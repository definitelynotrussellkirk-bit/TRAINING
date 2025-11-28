"""
Job Router - Find workers for jobs using the Device Registry.

Routes jobs to appropriate devices based on:
- Required device roles (eval_worker, data_forge, etc.)
- GPU requirements
- Current worker availability/load

Usage:
    from guild.job_router import JobRouter, get_router

    router = get_router()

    # Find workers for an eval job
    workers = router.find_workers(JobType.EVAL)

    # Get best worker (considering load)
    best = router.best_worker(JobType.EVAL)

    # Check if a job can be routed
    if router.can_route(job_spec):
        worker = router.route(job_spec)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from guild.job_types import JobType, JobSpec, JobPriority

logger = logging.getLogger("job_router")


@dataclass
class WorkerInfo:
    """
    Information about a worker for routing decisions.

    Combines device info with runtime status.
    """
    device_id: str
    hostname: str
    roles: List[str]
    has_gpu: bool
    vram_gb: float
    is_online: bool = True
    current_load: float = 0.0  # 0-1
    running_jobs: int = 0
    max_concurrent: int = 1
    capabilities: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Whether worker can accept new jobs."""
        return self.is_online and self.running_jobs < self.max_concurrent

    @property
    def available_slots(self) -> int:
        """Number of job slots available."""
        return max(0, self.max_concurrent - self.running_jobs)


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    success: bool
    worker: Optional[WorkerInfo] = None
    reason: str = ""
    alternatives: List[WorkerInfo] = field(default_factory=list)


class JobRouter:
    """
    Routes jobs to appropriate workers based on the Device Registry.

    Uses device roles and capabilities to match jobs to workers:
    - EVAL jobs → devices with EVAL_WORKER role + inference access
    - DATA_GEN jobs → devices with DATA_FORGE role
    - ARCHIVE jobs → devices with VAULT_WORKER role
    """

    # Map job types to required device roles
    JOB_TYPE_ROLES = {
        JobType.EVAL: ["eval_worker"],
        JobType.SPARRING: ["eval_worker"],  # Needs inference
        JobType.INFERENCE: ["inference"],
        JobType.DATA_GEN: ["data_forge"],
        JobType.DATA_FILTER: ["data_forge"],
        JobType.DATA_CONVERT: ["data_forge"],
        JobType.ARCHIVE: ["vault_worker"],
        JobType.RETENTION: ["vault_worker"],
        JobType.SYNC: ["vault_worker"],
        JobType.ANALYTICS: ["analytics"],
        JobType.REPORT: ["analytics"],
        JobType.HEALTH_CHECK: ["eval_worker", "data_forge", "vault_worker"],  # Any worker
    }

    def __init__(self):
        """Initialize the job router."""
        self._device_registry = None
        self._host_registry = None
        self._worker_status: Dict[str, WorkerInfo] = {}

    @property
    def device_registry(self):
        """Lazy load device registry."""
        if self._device_registry is None:
            from core.devices import get_device_registry
            self._device_registry = get_device_registry()
        return self._device_registry

    @property
    def host_registry(self):
        """Lazy load host registry."""
        if self._host_registry is None:
            from core.hosts import get_registry
            self._host_registry = get_registry()
        return self._host_registry

    def _device_to_worker_info(self, device) -> WorkerInfo:
        """Convert DeviceInfo to WorkerInfo."""
        # Check if we have cached status
        if device.device_id in self._worker_status:
            cached = self._worker_status[device.device_id]
            # Update device info but keep runtime status
            cached.hostname = device.hostname
            cached.roles = [r.value for r in device.roles]
            cached.has_gpu = device.has_gpu()
            cached.vram_gb = device.total_vram()
            return cached

        # Create new worker info
        worker = WorkerInfo(
            device_id=device.device_id,
            hostname=device.hostname,
            roles=[r.value for r in device.roles],
            has_gpu=device.has_gpu(),
            vram_gb=device.total_vram(),
            capabilities={
                "memory_gb": device.memory_gb,
                "storage_zones": device.storage_zones,
            },
        )

        # GPU devices can handle one job at a time
        # CPU workers can do more
        if worker.has_gpu:
            worker.max_concurrent = 1
        else:
            worker.max_concurrent = 3

        return worker

    def find_workers(
        self,
        job_type: JobType,
        require_gpu: Optional[bool] = None,
        require_available: bool = True,
    ) -> List[WorkerInfo]:
        """
        Find all workers capable of handling a job type.

        Args:
            job_type: Type of job to route
            require_gpu: Override GPU requirement
            require_available: Only return available workers

        Returns:
            List of capable workers, sorted by preference
        """
        from core.devices import DeviceRole

        # Get required roles for this job type
        required_roles = self.JOB_TYPE_ROLES.get(job_type, [])
        if not required_roles:
            logger.warning(f"No role mapping for job type: {job_type}")
            return []

        # Determine GPU requirement
        needs_gpu = require_gpu if require_gpu is not None else job_type.requires_gpu

        # Find devices with required roles
        candidates = []
        for role_str in required_roles:
            try:
                role = DeviceRole(role_str)
                devices = self.device_registry.devices_with_role(role)
                for device in devices:
                    if not device.enabled:
                        continue
                    worker = self._device_to_worker_info(device)

                    # Check GPU requirement
                    if needs_gpu and not worker.has_gpu:
                        continue

                    # Check availability
                    if require_available and not worker.is_available:
                        continue

                    if worker not in candidates:
                        candidates.append(worker)
            except ValueError:
                logger.warning(f"Unknown device role: {role_str}")

        # Sort by preference:
        # 1. Available slots (more is better)
        # 2. Lower current load
        # 3. Has GPU (for GPU jobs) or doesn't have GPU (for CPU jobs)
        def sort_key(w: WorkerInfo) -> Tuple:
            return (
                -w.available_slots,
                w.current_load,
                0 if (needs_gpu == w.has_gpu) else 1,
            )

        candidates.sort(key=sort_key)
        return candidates

    def best_worker(
        self,
        job_type: JobType,
        require_gpu: Optional[bool] = None,
    ) -> Optional[WorkerInfo]:
        """
        Get the best available worker for a job type.

        Args:
            job_type: Type of job to route
            require_gpu: Override GPU requirement

        Returns:
            Best worker or None if no workers available
        """
        workers = self.find_workers(job_type, require_gpu, require_available=True)
        return workers[0] if workers else None

    def route(self, spec: JobSpec) -> RoutingDecision:
        """
        Route a job spec to the best available worker.

        Args:
            spec: Job specification to route

        Returns:
            RoutingDecision with worker or failure reason
        """
        # Check for specific target device
        if spec.target_device:
            device = self.device_registry.get(spec.target_device)
            if device and device.enabled:
                worker = self._device_to_worker_info(device)
                if worker.is_available:
                    return RoutingDecision(
                        success=True,
                        worker=worker,
                        reason=f"Routed to specified target: {spec.target_device}",
                    )
                else:
                    return RoutingDecision(
                        success=False,
                        reason=f"Target device {spec.target_device} is not available",
                    )
            else:
                return RoutingDecision(
                    success=False,
                    reason=f"Target device {spec.target_device} not found or disabled",
                )

        # Find workers
        workers = self.find_workers(
            spec.job_type,
            require_gpu=spec.require_gpu,
            require_available=True,
        )

        if not workers:
            # Check if there are workers at all (just busy)
            all_workers = self.find_workers(
                spec.job_type,
                require_gpu=spec.require_gpu,
                require_available=False,
            )

            if all_workers:
                return RoutingDecision(
                    success=False,
                    reason=f"All {len(all_workers)} workers are busy",
                    alternatives=all_workers,
                )
            else:
                return RoutingDecision(
                    success=False,
                    reason=f"No workers configured for job type: {spec.job_type.value}",
                )

        # Return best worker
        return RoutingDecision(
            success=True,
            worker=workers[0],
            reason=f"Routed to {workers[0].device_id}",
            alternatives=workers[1:],
        )

    def can_route(self, spec: JobSpec) -> bool:
        """Check if a job can be routed to any worker."""
        decision = self.route(spec)
        return decision.success

    # =========================================================================
    # WORKER STATUS MANAGEMENT
    # =========================================================================

    def update_worker_status(
        self,
        device_id: str,
        is_online: bool = True,
        current_load: float = 0.0,
        running_jobs: int = 0,
    ) -> None:
        """
        Update runtime status of a worker.

        Called by worker heartbeats or status checks.
        """
        if device_id not in self._worker_status:
            device = self.device_registry.get(device_id)
            if device:
                self._worker_status[device_id] = self._device_to_worker_info(device)
            else:
                logger.warning(f"Unknown device: {device_id}")
                return

        worker = self._worker_status[device_id]
        worker.is_online = is_online
        worker.current_load = current_load
        worker.running_jobs = running_jobs

    def mark_job_started(self, device_id: str) -> None:
        """Mark that a job has started on a worker."""
        if device_id in self._worker_status:
            self._worker_status[device_id].running_jobs += 1

    def mark_job_completed(self, device_id: str) -> None:
        """Mark that a job has completed on a worker."""
        if device_id in self._worker_status:
            status = self._worker_status[device_id]
            status.running_jobs = max(0, status.running_jobs - 1)

    def get_worker_status(self, device_id: str) -> Optional[WorkerInfo]:
        """Get current status of a worker."""
        if device_id in self._worker_status:
            return self._worker_status[device_id]

        device = self.device_registry.get(device_id)
        if device:
            return self._device_to_worker_info(device)
        return None

    # =========================================================================
    # INFO
    # =========================================================================

    def get_routing_table(self) -> Dict[str, Any]:
        """Get routing table for all job types."""
        table = {}
        for job_type in JobType:
            workers = self.find_workers(job_type, require_available=False)
            available = [w for w in workers if w.is_available]

            table[job_type.value] = {
                "total_workers": len(workers),
                "available_workers": len(available),
                "workers": [
                    {
                        "device_id": w.device_id,
                        "hostname": w.hostname,
                        "is_available": w.is_available,
                        "has_gpu": w.has_gpu,
                        "running_jobs": w.running_jobs,
                    }
                    for w in workers
                ],
            }

        return table

    def get_summary(self) -> Dict[str, Any]:
        """Get router summary."""
        from core.devices import DeviceRole

        # Count workers by role
        role_counts = {}
        for role in [DeviceRole.EVAL_WORKER, DeviceRole.DATA_FORGE, DeviceRole.VAULT_WORKER]:
            devices = self.device_registry.devices_with_role(role)
            role_counts[role.value] = len(devices)

        return {
            "total_devices": len(self.device_registry.all_devices()),
            "enabled_devices": len(self.device_registry.enabled_devices()),
            "workers_by_role": role_counts,
            "tracked_workers": len(self._worker_status),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_router: Optional[JobRouter] = None


def get_router() -> JobRouter:
    """Get or create the job router singleton."""
    global _router
    if _router is None:
        _router = JobRouter()
    return _router


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Job Router - Find workers for jobs")
    parser.add_argument("command", nargs="?", default="table",
                        choices=["table", "route", "summary"])
    parser.add_argument("--job-type", help="Job type to route")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    router = get_router()

    if args.command == "table":
        table = router.get_routing_table()
        if args.json:
            print(json.dumps(table, indent=2))
        else:
            print("\n" + "=" * 70)
            print("JOB ROUTING TABLE")
            print("=" * 70)
            for job_type, info in table.items():
                workers = info["workers"]
                avail = info["available_workers"]
                total = info["total_workers"]
                print(f"\n{job_type}: {avail}/{total} workers available")
                for w in workers:
                    status = "✓" if w["is_available"] else "⏳"
                    gpu = "GPU" if w["has_gpu"] else "CPU"
                    print(f"  {status} {w['device_id']} ({w['hostname']}) [{gpu}]")
            print("\n" + "=" * 70)

    elif args.command == "route":
        if not args.job_type:
            print("Error: --job-type required for route command")
            exit(1)

        try:
            job_type = JobType(args.job_type)
        except ValueError:
            print(f"Unknown job type: {args.job_type}")
            print(f"Valid types: {[jt.value for jt in JobType]}")
            exit(1)

        spec = JobSpec(job_type=job_type)
        decision = router.route(spec)

        if args.json:
            print(json.dumps({
                "success": decision.success,
                "worker": decision.worker.device_id if decision.worker else None,
                "reason": decision.reason,
                "alternatives": [w.device_id for w in decision.alternatives],
            }, indent=2))
        else:
            if decision.success:
                w = decision.worker
                print(f"\n✅ Route {job_type.value} → {w.device_id}")
                print(f"   Host: {w.hostname}")
                print(f"   GPU: {w.has_gpu} ({w.vram_gb}GB)")
                if decision.alternatives:
                    print(f"   Alternatives: {[a.device_id for a in decision.alternatives]}")
            else:
                print(f"\n❌ Cannot route {job_type.value}")
                print(f"   Reason: {decision.reason}")

    elif args.command == "summary":
        summary = router.get_summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"\nJob Router Summary:")
            print(f"  Total devices: {summary['total_devices']}")
            print(f"  Enabled: {summary['enabled_devices']}")
            print(f"  Workers by role:")
            for role, count in summary['workers_by_role'].items():
                print(f"    {role}: {count}")
