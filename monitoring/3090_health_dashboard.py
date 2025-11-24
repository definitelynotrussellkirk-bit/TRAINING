#!/usr/bin/env python3
"""
3090 Health Dashboard - Comprehensive health monitoring and analytics
Provides detailed metrics, trends, and alerts for the remote GPU server
"""

import subprocess
import time
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
import statistics

class HealthDashboard:
    """Comprehensive health monitoring for 3090 server"""

    def __init__(
        self,
        remote_host: str = "192.168.x.x",
        api_port: int = 8765,
        base_dir: str = "/path/to/training"
    ):
        self.remote_host = remote_host
        self.api_port = api_port
        self.base_dir = Path(base_dir)

        # Metrics history (last 60 samples = 1 hour at 60s intervals)
        self.response_times = deque(maxlen=60)
        self.gpu_temps = deque(maxlen=60)
        self.vram_usage = deque(maxlen=60)
        self.gpu_utilization = deque(maxlen=60)
        self.error_count = deque(maxlen=60)

        # Daily stats
        self.daily_requests = 0
        self.daily_errors = 0
        self.daily_start = datetime.now()

    def get_api_metrics(self) -> Dict:
        """Get API performance metrics"""
        try:
            url = f"http://{self.remote_host}:{self.api_port}/health"
            start_time = time.time()
            response = requests.get(url, timeout=10)
            response_time = time.time() - start_time

            self.response_times.append(response_time)
            self.daily_requests += 1

            if response.status_code != 200:
                self.error_count.append(1)
                self.daily_errors += 1
            else:
                self.error_count.append(0)

            return {
                "status": "online" if response.status_code == 200 else "degraded",
                "response_time_ms": response_time * 1000,
                "status_code": response.status_code
            }
        except Exception as e:
            self.error_count.append(1)
            self.daily_errors += 1
            return {
                "status": "offline",
                "error": str(e)
            }

    def get_gpu_metrics(self) -> Dict:
        """Get GPU health metrics"""
        try:
            cmd = [
                "ssh", self.remote_host,
                "nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,"
                "utilization.gpu,utilization.memory,power.draw,power.limit "
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                temp = float(parts[0])
                mem_used = float(parts[1])
                mem_total = float(parts[2])
                gpu_util = float(parts[3])
                mem_util = float(parts[4])
                power_draw = float(parts[5])
                power_limit = float(parts[6])

                vram_pct = (mem_used / mem_total) * 100

                self.gpu_temps.append(temp)
                self.vram_usage.append(vram_pct)
                self.gpu_utilization.append(gpu_util)

                return {
                    "temperature_c": temp,
                    "vram_used_mb": mem_used,
                    "vram_total_mb": mem_total,
                    "vram_usage_pct": vram_pct,
                    "gpu_utilization_pct": gpu_util,
                    "memory_utilization_pct": mem_util,
                    "power_draw_w": power_draw,
                    "power_limit_w": power_limit,
                    "power_usage_pct": (power_draw / power_limit) * 100
                }
            else:
                return {"status": "error", "error": "nvidia-smi failed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_system_metrics(self) -> Dict:
        """Get system-level metrics (CPU, RAM, disk)"""
        try:
            # Get CPU usage
            cpu_cmd = ["ssh", self.remote_host, "top -bn1 | grep 'Cpu(s)' | awk '{print $2}'"]
            cpu_result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=10)
            cpu_usage = float(cpu_result.stdout.strip().replace("%us,", ""))

            # Get memory usage
            mem_cmd = ["ssh", self.remote_host, "free -m | awk 'NR==2{printf \"%.1f\", $3*100/$2 }'"]
            mem_result = subprocess.run(mem_cmd, capture_output=True, text=True, timeout=10)
            mem_usage = float(mem_result.stdout.strip())

            # Get disk usage
            disk_cmd = ["ssh", self.remote_host, "df -h ~/llm | awk 'NR==2{print $5}' | sed 's/%//'"]
            disk_result = subprocess.run(disk_cmd, capture_output=True, text=True, timeout=10)
            disk_usage = float(disk_result.stdout.strip())

            return {
                "cpu_usage_pct": cpu_usage,
                "ram_usage_pct": mem_usage,
                "disk_usage_pct": disk_usage
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def calculate_trends(self) -> Dict:
        """Calculate trends from historical data"""
        trends = {}

        if len(self.response_times) >= 10:
            trends["response_time"] = {
                "current_ms": self.response_times[-1] * 1000,
                "avg_ms": statistics.mean(self.response_times) * 1000,
                "p95_ms": statistics.quantiles(self.response_times, n=20)[18] * 1000,
                "trend": "improving" if self.response_times[-1] < statistics.mean(self.response_times) else "degrading"
            }

        if len(self.gpu_temps) >= 10:
            trends["temperature"] = {
                "current_c": self.gpu_temps[-1],
                "avg_c": statistics.mean(self.gpu_temps),
                "max_c": max(self.gpu_temps),
                "status": "normal" if max(self.gpu_temps) < 80 else "elevated"
            }

        if len(self.vram_usage) >= 10:
            trends["vram"] = {
                "current_pct": self.vram_usage[-1],
                "avg_pct": statistics.mean(self.vram_usage),
                "max_pct": max(self.vram_usage),
                "status": "normal" if max(self.vram_usage) < 90 else "high"
            }

        if len(self.error_count) >= 10:
            error_rate = sum(self.error_count) / len(self.error_count)
            trends["errors"] = {
                "error_rate_pct": error_rate * 100,
                "total_errors": sum(self.error_count),
                "status": "healthy" if error_rate < 0.05 else "degraded"
            }

        return trends

    def get_daily_stats(self) -> Dict:
        """Get daily statistics"""
        uptime = datetime.now() - self.daily_start

        return {
            "uptime_hours": uptime.total_seconds() / 3600,
            "total_requests": self.daily_requests,
            "total_errors": self.daily_errors,
            "error_rate_pct": (self.daily_errors / max(self.daily_requests, 1)) * 100,
            "requests_per_hour": self.daily_requests / max(uptime.total_seconds() / 3600, 1)
        }

    def generate_health_report(self) -> Dict:
        """Generate comprehensive health report"""
        api_metrics = self.get_api_metrics()
        gpu_metrics = self.get_gpu_metrics()
        system_metrics = self.get_system_metrics()
        trends = self.calculate_trends()
        daily_stats = self.get_daily_stats()

        # Overall health score (0-100)
        health_score = 100

        if api_metrics["status"] == "offline":
            health_score -= 50
        elif api_metrics["status"] == "degraded":
            health_score -= 25

        if gpu_metrics.get("temperature_c", 0) > 80:
            health_score -= 15

        if gpu_metrics.get("vram_usage_pct", 0) > 90:
            health_score -= 10

        if daily_stats["error_rate_pct"] > 5:
            health_score -= 20

        health_status = "excellent" if health_score >= 90 else \
                       "good" if health_score >= 75 else \
                       "fair" if health_score >= 50 else \
                       "poor"

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": {
                "score": max(health_score, 0),
                "status": health_status
            },
            "api": api_metrics,
            "gpu": gpu_metrics,
            "system": system_metrics,
            "trends": trends,
            "daily_stats": daily_stats
        }

    def print_dashboard(self, report: Dict):
        """Print formatted dashboard to console"""
        print("\n" + "=" * 80)
        print("🖥️  3090 HEALTH DASHBOARD")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Health: {report['overall_health']['status'].upper()} "
              f"(Score: {report['overall_health']['score']}/100)")
        print("-" * 80)

        # API Status
        api = report['api']
        print("📡 API STATUS")
        print(f"  Status: {api['status'].upper()}")
        if 'response_time_ms' in api:
            print(f"  Response Time: {api['response_time_ms']:.1f}ms")
        print()

        # GPU Status
        gpu = report['gpu']
        if 'error' not in gpu:
            print("🎮 GPU STATUS")
            print(f"  Temperature: {gpu['temperature_c']:.1f}°C")
            print(f"  VRAM Usage: {gpu['vram_used_mb']:.0f}MB / {gpu['vram_total_mb']:.0f}MB "
                  f"({gpu['vram_usage_pct']:.1f}%)")
            print(f"  GPU Utilization: {gpu['gpu_utilization_pct']:.1f}%")
            print(f"  Power Draw: {gpu['power_draw_w']:.1f}W / {gpu['power_limit_w']:.1f}W "
                  f"({gpu['power_usage_pct']:.1f}%)")
            print()

        # System Status
        sys = report['system']
        if 'error' not in sys:
            print("💻 SYSTEM STATUS")
            print(f"  CPU Usage: {sys['cpu_usage_pct']:.1f}%")
            print(f"  RAM Usage: {sys['ram_usage_pct']:.1f}%")
            print(f"  Disk Usage: {sys['disk_usage_pct']:.1f}%")
            print()

        # Trends
        if report['trends']:
            print("📊 TRENDS (Last Hour)")
            if 'response_time' in report['trends']:
                rt = report['trends']['response_time']
                print(f"  Response Time: {rt['current_ms']:.1f}ms (avg: {rt['avg_ms']:.1f}ms, "
                      f"p95: {rt['p95_ms']:.1f}ms)")
            if 'temperature' in report['trends']:
                temp = report['trends']['temperature']
                print(f"  Temperature: {temp['current_c']:.1f}°C (avg: {temp['avg_c']:.1f}°C, "
                      f"max: {temp['max_c']:.1f}°C)")
            if 'errors' in report['trends']:
                err = report['trends']['errors']
                print(f"  Error Rate: {err['error_rate_pct']:.1f}% ({err['total_errors']} errors)")
            print()

        # Daily Stats
        daily = report['daily_stats']
        print("📈 DAILY STATISTICS")
        print(f"  Uptime: {daily['uptime_hours']:.1f} hours")
        print(f"  Total Requests: {daily['total_requests']}")
        print(f"  Total Errors: {daily['total_errors']} ({daily['error_rate_pct']:.1f}%)")
        print(f"  Request Rate: {daily['requests_per_hour']:.1f}/hour")

        print("=" * 80)

    def run_monitoring(self, interval: int = 60, save_reports: bool = True):
        """Run continuous monitoring"""
        reports_dir = self.base_dir / "logs" / "3090_health"
        reports_dir.mkdir(parents=True, exist_ok=True)

        print("🚀 Starting 3090 Health Dashboard")
        print(f"Monitoring interval: {interval}s")
        print(f"Reports directory: {reports_dir}")
        print("\nPress Ctrl+C to stop\n")

        while True:
            try:
                report = self.generate_health_report()
                self.print_dashboard(report)

                if save_reports:
                    # Save to JSON
                    report_file = reports_dir / f"health_{datetime.now().strftime('%Y%m%d')}.json"

                    # Load existing reports
                    if report_file.exists():
                        with report_file.open() as f:
                            reports = json.load(f)
                    else:
                        reports = []

                    reports.append(report)

                    # Keep last 1440 reports (24 hours at 60s intervals)
                    reports = reports[-1440:]

                    with report_file.open('w') as f:
                        json.dump(reports, f, indent=2)

                time.sleep(interval)

            except KeyboardInterrupt:
                print("\n\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(interval)


def main():
    dashboard = HealthDashboard(
        remote_host="192.168.x.x",
        api_port=8765,
        base_dir="/path/to/training"
    )

    # Run monitoring every 60 seconds
    dashboard.run_monitoring(interval=60, save_reports=True)


if __name__ == "__main__":
    main()
