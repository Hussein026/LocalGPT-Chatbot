#!/usr/bin/env python3
"""
RAG System Unified Launcher (Windows-friendly)
=============================================

Starts:
- Ollama server (11434)
- Backend server (8000)
- Frontend (3000, optional)

Notes
-----
• RAG API (8001 / LanceDB) is DISABLED on Windows by default because LanceDB
  wheels are not supported; attempting to load _lancedb raises a DLL error.
• You can also disable RAG anywhere with --no-rag.

Usage:
    python run_system.py [--mode dev|prod] [--logs-only] [--no-frontend] [--no-rag]
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import requests


# --------------------------- Config & helpers ---------------------------

@dataclass
class ServiceConfig:
    name: str
    command: List[str]
    port: int
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    health_check_path: str = "/health"
    startup_delay: int = 2
    required: bool = True


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    SERVICE_COLORS = {
        'ollama': '\033[94m',
        'rag-api': '\033[95m',
        'backend': '\033[96m',
        'frontend': '\033[93m',
        'system': '\033[92m',
    }
    RESET = '\033[0m'

    def format(self, record):
        service_name = getattr(record, 'service', 'system')
        service_color = self.SERVICE_COLORS.get(service_name, '')
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        colored_service = f"{service_color}[{service_name.upper()}]{self.RESET}"
        colored_level = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{self.RESET}"
        return f"{timestamp} {colored_service} {colored_level}: {record.getMessage()}"


class ServiceManager:
    def __init__(self, mode: str = "dev", logs_dir: str = "logs", no_frontend: bool = False, no_rag: bool = False):
        self.mode = mode
        self.no_frontend = no_frontend
        self.no_rag = no_rag or (os.name == "nt")  # auto-disable RAG on Windows
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)

        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_threads: Dict[str, threading.Thread] = {}
        self.running = False

        self._setup_logging()
        self.services = self._get_service_configs()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        if os.name == "nt":
            self.logger.info("ℹ️  Windows detected: RAG API (LanceDB) will be disabled automatically.")
        if self.no_rag:
            self.logger.info("ℹ️  RAG API disabled (no_rag=True). System will run in chat/document fallback mode.")

    # --------------------------- logging ---------------------------

    def _setup_logging(self):
        self.logger = logging.getLogger('system')
        self.logger.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(ColoredFormatter())
        self.logger.addHandler(ch)

        fh = logging.FileHandler(self.logs_dir / 'system.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(fh)

    # --------------------------- service configs ---------------------------

    def _get_service_configs(self) -> Dict[str, ServiceConfig]:
        base = {}

        base['ollama'] = ServiceConfig(
            name='ollama',
            command=['ollama', 'serve'],
            port=11434,
            startup_delay=5,
            required=True
        )

        # RAG API – include only if enabled
        if not self.no_rag:
            base['rag-api'] = ServiceConfig(
                name='rag-api',
                command=[sys.executable, '-m', 'rag_system.api_server'],
                port=8001,
                startup_delay=3,
                required=True
            )

        base['backend'] = ServiceConfig(
            name='backend',
            command=[sys.executable, 'backend/server.py'],
            port=8000,
            startup_delay=2,
            required=True
        )

        base['frontend'] = ServiceConfig(
            name='frontend',
            command=['npm', 'run', 'dev' if self.mode == 'dev' else 'start'],
            port=3000,
            startup_delay=5,
            required=False
        )

        if self.mode == 'prod':
            base['frontend'].command = ['npm', 'run', 'start']
            base['backend'].env = {'NODE_ENV': 'production'}

        return base

    # --------------------------- utils ---------------------------

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)

    def _command_exists(self, cmd: str) -> bool:
        try:
            subprocess.run([cmd, '--version'], capture_output=True, check=True, timeout=5)
            return True
        except Exception:
            return False

    def is_port_in_use(self, port: int) -> bool:
        try:
            for c in psutil.net_connections():
                if c.laddr and c.laddr.port == port and c.status == 'LISTEN':
                    return True
            return False
        except Exception:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('127.0.0.1', port)) == 0

    def check_prerequisites(self) -> bool:
        self.logger.info("🔍 Checking prerequisites...")
        missing = []

        if not self._command_exists('ollama'):
            missing.append('ollama')

        if not (self._command_exists('python') or self._command_exists('python3')):
            missing.append('python')

        if not self._command_exists('npm'):
            self.logger.warning("⚠️  npm not found - frontend will be disabled")
            self.services.get('frontend', ServiceConfig('frontend', [], 3000)).required = False
            self.no_frontend = True

        if missing:
            self.logger.error(f"❌ Missing required tools: {', '.join(missing)}")
            return False

        self.logger.info("✅ All prerequisites satisfied")
        return True

    # --------------------------- model ensure ---------------------------

    def ensure_models(self):
        self.logger.info("📥 Checking required models...")
        required_models = ['qwen3:8b', 'qwen3:0.6b']
        try:
            out = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10).stdout
            for m in required_models:
                if m not in out:
                    self.logger.info(f"📥 Pulling {m}...")
                    subprocess.run(['ollama', 'pull', m], check=True)
                    self.logger.info(f"✅ {m} ready")
                else:
                    self.logger.info(f"✅ {m} already available")
        except Exception as e:
            self.logger.warning(f"⚠️  Model check failed: {e}")

    # --------------------------- service start/stop ---------------------------

    def start_service(self, name: str, cfg: ServiceConfig) -> bool:
        if name in self.processes:
            self.logger.warning(f"⚠️  {name} already running")
            return True

        if self.is_port_in_use(cfg.port):
            self.logger.info(f"ℹ️  Port {cfg.port} already in use, assuming {name} is running")
            return True

        self.logger.info(f"🔄 Starting {name} on port {cfg.port}...")
        env = os.environ.copy()
        if cfg.env:
            env.update(cfg.env)

        try:
            p = subprocess.Popen(
                cfg.command,
                cwd=cfg.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            self.processes[name] = p

            t = threading.Thread(target=self._monitor_logs, args=(name, p), daemon=True)
            t.start()

            time.sleep(cfg.startup_delay)
            if p.poll() is None:
                self.logger.info(f"✅ {name} started successfully (PID: {p.pid})")
                return True
            else:
                self.logger.error(f"❌ {name} failed to start")
                return False
        except Exception as e:
            self.logger.error(f"❌ Failed to start {name}: {e}")
            return False

    def _monitor_logs(self, name: str, proc: subprocess.Popen):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.logs_dir / f'{name}.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(fh)

        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            msg = line.strip()
            if not msg:
                continue
            rec = logging.LogRecord(name=name, level=logging.INFO, pathname='', lineno=0, msg=msg, args=(), exc_info=None)
            rec.service = name
            logger.handle(rec)
            logging.getLogger('system').handle(rec)

    def _start_ollama(self) -> bool:
        if self.is_port_in_use(11434):
            self.logger.info("✅ Ollama already running")
            self.ensure_models()
            return True
        if self.start_service('ollama', self.services['ollama']):
            self.ensure_models()
            return True
        return False

    def start_all(self) -> bool:
        self.logger.info("🚀 Starting system components...")
        if not self.check_prerequisites():
            return False

        self.running = True
        order: List[str] = ['ollama']
        if not self.no_rag and 'rag-api' in self.services:
            order.append('rag-api')
        order.append('backend')
        if not self.no_frontend and 'frontend' in self.services:
            order.append('frontend')

        failed: List[str] = []
        for name in order:
            cfg = self.services[name]
            if name == 'ollama':
                ok = self._start_ollama()
            else:
                ok = self.start_service(name, cfg)
            if not ok and cfg.required:
                failed.append(name)

        if failed:
            self.logger.error(f"❌ Failed to start required services: {', '.join(failed)}")
            return False

        self._print_summary()
        return True

    def _print_summary(self):
        self.logger.info("")
        self.logger.info("🎉 System Started!")
        for name, cfg in self.services.items():
            status = "Running" if (name in self.processes or self.is_port_in_use(cfg.port)) else "Stopped"
            url = f"http://127.0.0.1:{cfg.port}"
            self.logger.info(f" • {name:<9} : {'✅' if status=='Running' else '❌'} {status:<8} {url}")
        self.logger.info("")
        self.logger.info("Frontend: http://127.0.0.1:3000")
        self.logger.info("Backend : http://127.0.0.1:8000")
        if not self.no_rag:
            self.logger.info("RAG API : http://127.0.0.1:8001")
        else:
            self.logger.info("RAG API : disabled")
        self.logger.info("")

    def shutdown(self):
        if not self.running:
            return
        self.logger.info("🛑 Shutting down...")
        self.running = False
        for name, proc in list(self.processes.items())[::-1]:
            self._stop_service(name)

    def _stop_service(self, name: str):
        proc = self.processes.get(name)
        if not proc:
            return
        self.logger.info(f"🔄 Stopping {name}...")
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            self.logger.info(f"✅ {name} stopped")
        finally:
            self.processes.pop(name, None)

    def monitor(self):
        self.logger.info("👁️  Monitoring... (Ctrl+C to stop)")
        try:
            while self.running:
                time.sleep(30)
                for name, proc in list(self.processes.items()):
                    if proc.poll() is not None:
                        self.logger.warning(f"⚠️  {name} exited; restarting...")
                        cfg = self.services[name]
                        self.start_service(name, cfg)
        except KeyboardInterrupt:
            pass


# --------------------------- CLI ---------------------------

def main():
    p = argparse.ArgumentParser(description="Unified launcher")
    p.add_argument('--mode', choices=['dev', 'prod'], default='dev')
    p.add_argument('--logs-only', action='store_true')
    p.add_argument('--no-frontend', action='store_true', help='Skip frontend startup')
    p.add_argument('--no-rag', action='store_true', help='Disable RAG API (also auto-disabled on Windows)')
    p.add_argument('--health', action='store_true')
    p.add_argument('--stop', action='store_true')

    args = p.parse_args()

    mgr = ServiceManager(mode=args.mode, no_frontend=args.no_frontend, no_rag=args.no_rag)

    try:
        if args.health:
            mgr._print_summary()
            return
        if args.stop:
            mgr.shutdown()
            return
        if args.logs_only:
            mgr.monitor()
            return

        if mgr.start_all():
            mgr.monitor()
        else:
            mgr.logger.error("❌ System startup failed")
            sys.exit(1)
    finally:
        mgr.shutdown()


if __name__ == "__main__":
    main()
