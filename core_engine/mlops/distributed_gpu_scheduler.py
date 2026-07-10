"""
Distributed GPU Scheduler for NEXUS-OMEGA Stage 16

Manages GPU resource allocation across multiple retraining jobs
with strict quota enforcement to prevent starvation of live trading.
"""

import ray
from ray import remote
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
import time
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Priority levels for training jobs"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class GPUResource:
    """Represents a GPU resource"""
    gpu_id: int
    total_memory_mb: int
    allocated_memory_mb: int = 0
    current_job_id: Optional[str] = None
    utilization_percent: float = 0.0
    
    def available_memory(self) -> int:
        return self.total_memory_mb - self.allocated_memory_mb
    
    def is_available(self) -> bool:
        return self.current_job_id is None


@dataclass
class TrainingJob:
    """Represents a training job request"""
    job_id: str
    model_id: str
    priority: JobPriority
    requested_gpus: int
    requested_memory_per_gpu: int
    max_duration_minutes: int
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"


@remote
class DistributedGPUScheduler:
    """
    Distributes GPU resources across training jobs with:
    - Strict quota enforcement
    - Priority-based scheduling
    - Preemption support for critical jobs
    - Fair-share allocation
    """
    
    def __init__(
        self,
        total_gpus: int = 8,
        gpu_memory_mb: int = 16384,
        max_training_quota: float = 0.8,
        enable_preemption: bool = True
    ):
        self.total_gpus = total_gpus
        self.gpu_memory_mb = gpu_memory_mb
        
        # Initialize GPU resources
        self.gpus: List[GPUResource] = [
            GPUResource(gpu_id=i, total_memory_mb=gpu_memory_mb)
            for i in range(total_gpus)
        ]
        
        # Job queues by priority
        self.job_queues: Dict[JobPriority, List[TrainingJob]] = {
            p: [] for p in JobPriority
        }
        
        # Active jobs
        self.active_jobs: Dict[str, TrainingJob] = {}
        
        # Completed jobs history
        self.completed_jobs: List[TrainingJob] = []
        
        # Resource quotas
        self.max_training_quota = max_training_quota
        self.max_training_gpus = int(total_gpus * max_training_quota)
        self.enable_preemption = enable_preemption
        
        # Statistics
        self.stats = {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_preempted": 0,
            "total_gpu_hours": 0.0,
        }
    
    def submit_job(
        self,
        job_id: str,
        model_id: str,
        priority: str = "MEDIUM",
        requested_gpus: int = 1,
        requested_memory_mb: int = 4096,
        max_duration_minutes: int = 60,
    ) -> bool:
        """Submit a new training job"""
        try:
            job_priority = JobPriority[priority.upper()]
        except KeyError:
            job_priority = JobPriority.MEDIUM
        
        job = TrainingJob(
            job_id=job_id,
            model_id=model_id,
            priority=job_priority,
            requested_gpus=min(requested_gpus, self.max_training_gpus),
            requested_memory_per_gpu=min(requested_memory_mb, self.gpu_memory_mb),
            max_duration_minutes=max_duration_minutes,
        )
        
        # Add to appropriate queue
        self.job_queues[job.priority].append(job)
        self.stats["jobs_submitted"] += 1
        
        logger.info(f"Submitted job {job_id} with priority {job.priority.name}")
        
        # Try to schedule immediately
        return self._try_schedule_jobs()
    
    def _try_schedule_jobs(self) -> bool:
        """Try to schedule pending jobs"""
        scheduled_any = False
        
        # Process queues in priority order
        for priority in sorted(JobPriority, key=lambda p: p.value, reverse=True):
            queue = self.job_queues[priority]
            
            while queue:
                job = queue[0]
                
                # Check if we have enough resources
                if self._can_schedule_job(job):
                    # Schedule the job
                    self._schedule_job(job)
                    queue.pop(0)
                    scheduled_any = True
                else:
                    # Not enough resources, check preemption for high priority
                    if priority == JobPriority.CRITICAL and self.enable_preemption:
                        if self._try_preempt_for_job(job):
                            continue
                    break
        
        return scheduled_any
    
    def _can_schedule_job(self, job: TrainingJob) -> bool:
        """Check if job can be scheduled with current resources"""
        # Count available GPUs with sufficient memory
        available_gpus = sum(
            1 for gpu in self.gpus
            if gpu.is_available() and gpu.available_memory() >= job.requested_memory_per_gpu
        )
        
        # Check total training GPU limit
        active_training_gpus = sum(
            1 for gpu in self.gpus
            if gpu.current_job_id in self.active_jobs
        )
        
        return (
            available_gpus >= job.requested_gpus and
            active_training_gpus + job.requested_gpus <= self.max_training_gpus
        )
    
    def _schedule_job(self, job: TrainingJob):
        """Schedule a job on available GPUs"""
        job.started_at = time.time()
        job.status = "running"
        
        # Find and allocate GPUs
        allocated = 0
        for gpu in self.gpus:
            if allocated >= job.requested_gpus:
                break
            
            if gpu.is_available() and gpu.available_memory() >= job.requested_memory_per_gpu:
                gpu.allocated_memory_mb = job.requested_memory_per_gpu
                gpu.current_job_id = job.job_id
                allocated += 1
        
        self.active_jobs[job.job_id] = job
        
        logger.info(f"Scheduled job {job.job_id} on {allocated} GPUs")
    
    def _try_preempt_for_job(self, job: TrainingJob) -> bool:
        """Try to preempt lower priority jobs for critical job"""
        # Find lowest priority active job
        lowest_priority_job = None
        lowest_priority = JobPriority.CRITICAL
        
        for active_job in self.active_jobs.values():
            if active_job.priority.value < lowest_priority.value:
                lowest_priority = active_job.priority
                lowest_priority_job = active_job
        
        # Preempt if found and lower priority than new job
        if lowest_priority_job and lowest_priority_job.priority.value < job.priority.value:
            logger.info(
                f"Preempting job {lowest_priority_job.job_id} for critical job {job.job_id}"
            )
            self._preempt_job(lowest_priority_job.job_id)
            self.stats["jobs_preempted"] += 1
            return True
        
        return False
    
    def _preempt_job(self, job_id: str):
        """Preempt a running job"""
        if job_id not in self.active_jobs:
            return
        
        job = self.active_jobs[job_id]
        job.status = "preempted"
        
        # Release GPUs
        for gpu in self.gpus:
            if gpu.current_job_id == job_id:
                gpu.allocated_memory_mb = 0
                gpu.current_job_id = None
        
        # Re-queue the job
        self.job_queues[job.priority].insert(0, job)
        
        del self.active_jobs[job_id]
    
    def complete_job(self, job_id: str, success: bool = True) -> bool:
        """Mark a job as completed"""
        if job_id not in self.active_jobs:
            return False
        
        job = self.active_jobs[job_id]
        job.completed_at = time.time()
        job.status = "completed" if success else "failed"
        
        # Release GPUs
        for gpu in self.gpus:
            if gpu.current_job_id == job_id:
                gpu.allocated_memory_mb = 0
                gpu.current_job_id = None
                gpu.utilization_percent = 0.0
        
        # Update statistics
        if job.started_at and job.completed_at:
            duration_hours = (job.completed_at - job.started_at) / 3600
            self.stats["total_gpu_hours"] += duration_hours * job.requested_gpus
        
        self.stats["jobs_completed"] += 1
        
        # Move to completed history
        self.completed_jobs.append(job)
        del self.active_jobs[job_id]
        
        logger.info(f"Completed job {job_id} (success={success})")
        
        # Try to schedule more jobs
        self._try_schedule_jobs()
        
        return True
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a specific job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                "job_id": job.job_id,
                "status": job.status,
                "priority": job.priority.name,
                "gpus_allocated": job.requested_gpus,
                "started_at": job.started_at,
                "duration_seconds": time.time() - job.started_at if job.started_at else None,
            }
        
        # Check pending queues
        for priority, queue in self.job_queues.items():
            for job in queue:
                if job.job_id == job_id:
                    return {
                        "job_id": job.job_id,
                        "status": "pending",
                        "priority": priority.name,
                        "position_in_queue": queue.index(job) + 1,
                    }
        
        # Check completed
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return {
                    "job_id": job.job_id,
                    "status": job.status,
                    "priority": job.priority.name,
                    "completed_at": job.completed_at,
                }
        
        return {"error": "Job not found"}
    
    def get_cluster_status(self) -> Dict:
        """Get overall cluster status"""
        active_count = len(self.active_jobs)
        pending_count = sum(len(q) for q in self.job_queues.values())
        
        gpu_status = []
        for gpu in self.gpus:
            gpu_status.append({
                "gpu_id": gpu.gpu_id,
                "available_memory_mb": gpu.available_memory(),
                "current_job": gpu.current_job_id,
                "utilization": gpu.utilization_percent,
            })
        
        return {
            "total_gpus": self.total_gpus,
            "active_gpus": sum(1 for g in self.gpus if g.current_job_id),
            "max_training_gpus": self.max_training_gpus,
            "active_jobs": active_count,
            "pending_jobs": pending_count,
            "gpu_details": gpu_status,
            "statistics": self.stats.copy(),
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job"""
        # Check active jobs
        if job_id in self.active_jobs:
            self.complete_job(job_id, success=False)
            return True
        
        # Check pending queues
        for priority, queue in self.job_queues.items():
            for i, job in enumerate(queue):
                if job.job_id == job_id:
                    queue.pop(i)
                    job.status = "cancelled"
                    self.completed_jobs.append(job)
                    return True
        
        return False
    
    def update_gpu_utilization(self, gpu_id: int, utilization: float):
        """Update GPU utilization metric"""
        if 0 <= gpu_id < len(self.gpus):
            self.gpus[gpu_id].utilization_percent = min(100.0, max(0.0, utilization))
    
    def get_pending_time_estimate(self, priority: str, requested_gpus: int) -> float:
        """Estimate wait time for a job with given requirements"""
        try:
            job_priority = JobPriority[priority.upper()]
        except KeyError:
            job_priority = JobPriority.MEDIUM
        
        # Count higher priority pending jobs
        higher_priority_count = 0
        for p in JobPriority:
            if p.value > job_priority.value:
                higher_priority_count += len(self.job_queues[p])
        
        # Estimate based on average job duration
        if self.stats["jobs_completed"] > 0:
            avg_duration = self.stats["total_gpu_hours"] / self.stats["jobs_completed"]
        else:
            avg_duration = 0.5  # Default 30 minutes
        
        # Rough estimate
        estimated_wait = higher_priority_count * avg_duration / max(1, self.total_gpus - len(self.active_jobs))
        
        return estimated_wait


async def main():
    """Example usage"""
    scheduler = DistributedGPUScheduler.remote(
        total_gpus=8,
        gpu_memory_mb=16384,
        max_training_quota=0.75,
    )
    
    # Submit some jobs
    scheduler.submit_job.remote("job1", "model_a", "LOW", 2, 4096, 30)
    scheduler.submit_job.remote("job2", "model_b", "HIGH", 4, 8192, 60)
    scheduler.submit_job.remote("job3", "model_c", "CRITICAL", 2, 4096, 15)
    
    # Check status
    status = await scheduler.get_cluster_status.remote()
    print(f"Cluster status: {status}")
    
    # Complete a job
    scheduler.complete_job.remote("job1", success=True)
    
    # Check again
    status = await scheduler.get_cluster_status.remote()
    print(f"After completion: {status}")


if __name__ == "__main__":
    ray.init()
    asyncio.run(main())
    ray.shutdown()
