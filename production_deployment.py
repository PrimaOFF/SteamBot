#!/usr/bin/env python3

import asyncio
import logging
import os
import subprocess
import json
import time
import psutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import docker
import yaml

@dataclass
class HealthCheck:
    """Health check result"""
    service_name: str
    status: str  # "healthy", "unhealthy", "unknown"
    response_time: float
    last_check: datetime
    error_message: Optional[str] = None

@dataclass
class DeploymentStatus:
    """Deployment status information"""
    environment: str
    version: str
    status: str  # "deploying", "healthy", "failed", "rollback"
    services: Dict[str, HealthCheck]
    deployment_start: datetime
    deployment_duration: Optional[float] = None
    rollback_version: Optional[str] = None

class ProductionDeployment:
    """Production deployment and monitoring system"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        self.docker_client = docker.from_env()
        
        # Deployment configuration
        self.config = {
            'health_check_timeout': 30,
            'max_deployment_time': 600,  # 10 minutes
            'rollback_on_failure': True,
            'backup_before_deploy': True,
            'notification_channels': ['telegram', 'slack'],
            'required_services': [
                'cs2-float-checker',
                'postgres',
                'redis',
                'nginx'
            ]
        }
        
        # Service health endpoints
        self.health_endpoints = {
            'cs2-float-checker': 'http://localhost:8000/health',
            'postgres': 'postgresql://localhost:5432/cs2_trading',
            'redis': 'redis://localhost:6379',
            'nginx': 'http://localhost:80/health'
        }
        
        # Monitoring metrics
        self.metrics = {
            'deployment_count': 0,
            'successful_deployments': 0,
            'failed_deployments': 0,
            'average_deployment_time': 0.0,
            'rollback_count': 0,
            'last_deployment': None
        }
    
    async def deploy(self, version: str, dry_run: bool = False) -> DeploymentStatus:
        """Deploy application to production"""
        self.logger.info(f"🚀 Starting deployment of version {version} to {self.environment}")
        
        deployment_start = datetime.now()
        status = DeploymentStatus(
            environment=self.environment,
            version=version,
            status="deploying",
            services={},
            deployment_start=deployment_start
        )
        
        try:
            if dry_run:
                self.logger.info("🧪 DRY RUN: Simulating deployment...")
                await self._simulate_deployment(status)
                return status
            
            # Pre-deployment checks
            await self._pre_deployment_checks(status)
            
            # Backup current state
            if self.config['backup_before_deploy']:
                await self._backup_data(status)
            
            # Rolling deployment
            await self._rolling_deployment(status, version)
            
            # Post-deployment validation
            await self._post_deployment_validation(status)
            
            # Update metrics
            self._update_deployment_metrics(True, deployment_start)
            
            status.status = "healthy"
            status.deployment_duration = (datetime.now() - deployment_start).total_seconds()
            
            self.logger.info(f"✅ Deployment completed successfully in {status.deployment_duration:.1f} seconds")
            
        except Exception as e:
            self.logger.error(f"❌ Deployment failed: {e}")
            status.status = "failed"
            
            if self.config['rollback_on_failure']:
                await self._rollback_deployment(status)
            
            self._update_deployment_metrics(False, deployment_start)
            raise
        
        finally:
            # Send notifications
            await self._send_deployment_notifications(status)
        
        return status
    
    async def _pre_deployment_checks(self, status: DeploymentStatus):
        """Perform pre-deployment checks"""
        self.logger.info("🔍 Running pre-deployment checks...")
        
        # Check system resources
        await self._check_system_resources()
        
        # Check current service health
        current_health = await self._check_all_services_health()
        if not all(check.status == "healthy" for check in current_health.values()):
            unhealthy_services = [name for name, check in current_health.items() 
                                if check.status != "healthy"]
            raise Exception(f"Pre-deployment check failed: Unhealthy services: {unhealthy_services}")
        
        # Check Docker daemon
        try:
            self.docker_client.ping()
        except Exception as e:
            raise Exception(f"Docker daemon not accessible: {e}")
        
        # Check available disk space
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > 90:
            raise Exception(f"Insufficient disk space: {disk_usage.percent}% used")
        
        self.logger.info("✅ Pre-deployment checks passed")
    
    async def _backup_data(self, status: DeploymentStatus):
        """Backup critical data before deployment"""
        self.logger.info("💾 Creating backup before deployment...")
        
        backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = Path(f"/backups/{backup_timestamp}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Backup PostgreSQL database
            postgres_backup = backup_dir / "postgres_backup.sql"
            subprocess.run([
                'docker-compose', 'exec', '-T', 'postgres',
                'pg_dump', '-U', 'cs2user', 'cs2_trading'
            ], stdout=open(postgres_backup, 'w'), check=True)
            
            # Backup Redis data
            redis_backup = backup_dir / "redis_backup.rdb"
            subprocess.run([
                'docker-compose', 'exec', '-T', 'redis',
                'redis-cli', 'BGSAVE'
            ], check=True)
            
            # Copy Redis dump
            subprocess.run([
                'docker', 'cp', 'cs2-redis:/data/dump.rdb', str(redis_backup)
            ], check=True)
            
            # Backup application data
            subprocess.run([
                'cp', '-r', './data', str(backup_dir / 'app_data')
            ], check=True)
            
            status.rollback_version = backup_timestamp
            self.logger.info(f"✅ Backup created: {backup_dir}")
            
        except Exception as e:
            raise Exception(f"Backup failed: {e}")
    
    async def _rolling_deployment(self, status: DeploymentStatus, version: str):
        """Perform rolling deployment"""
        self.logger.info("🔄 Starting rolling deployment...")
        
        # Pull new images
        self.logger.info("📥 Pulling new Docker images...")
        subprocess.run(['docker-compose', 'pull'], check=True)
        
        # Update services one by one
        services_order = [
            'cs2-float-checker',
            'celery-worker', 
            'celery-beat',
            'nginx'
        ]
        
        for service in services_order:
            self.logger.info(f"🔄 Updating service: {service}")
            
            # Scale down old instance
            subprocess.run([
                'docker-compose', 'up', '-d', '--no-deps', '--scale', f'{service}=0', service
            ], check=True)
            
            # Start new instance
            subprocess.run([
                'docker-compose', 'up', '-d', '--no-deps', service
            ], check=True)
            
            # Health check
            await asyncio.sleep(10)  # Wait for service to start
            health_check = await self._check_service_health(service)
            
            if health_check.status != "healthy":
                raise Exception(f"Service {service} failed health check after update")
            
            status.services[service] = health_check
            self.logger.info(f"✅ Service {service} updated successfully")
    
    async def _post_deployment_validation(self, status: DeploymentStatus):
        """Validate deployment success"""
        self.logger.info("🔍 Running post-deployment validation...")
        
        # Wait for all services to stabilize
        await asyncio.sleep(30)
        
        # Check all services
        all_healthy = True
        for service in self.config['required_services']:
            health_check = await self._check_service_health(service)
            status.services[service] = health_check
            
            if health_check.status != "healthy":
                all_healthy = False
                self.logger.error(f"❌ Service {service} is not healthy: {health_check.error_message}")
        
        if not all_healthy:
            raise Exception("Post-deployment validation failed: Some services are unhealthy")
        
        # Run integration tests
        await self._run_integration_tests()
        
        self.logger.info("✅ Post-deployment validation passed")
    
    async def _check_service_health(self, service_name: str) -> HealthCheck:
        """Check health of a specific service"""
        start_time = time.time()
        
        try:
            if service_name == 'cs2-float-checker':
                # HTTP health check
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.health_endpoints[service_name],
                        timeout=aiohttp.ClientTimeout(total=self.config['health_check_timeout'])
                    ) as response:
                        if response.status == 200:
                            status = "healthy"
                            error_message = None
                        else:
                            status = "unhealthy"
                            error_message = f"HTTP {response.status}"
            
            elif service_name == 'postgres':
                # Database connection check
                import asyncpg
                try:
                    conn = await asyncpg.connect(
                        host='localhost',
                        port=5432,
                        user='cs2user',
                        password='cs2password',
                        database='cs2_trading'
                    )
                    await conn.execute('SELECT 1')
                    await conn.close()
                    status = "healthy"
                    error_message = None
                except Exception as e:
                    status = "unhealthy"
                    error_message = str(e)
            
            elif service_name == 'redis':
                # Redis connection check
                import aioredis
                try:
                    redis = aioredis.from_url('redis://localhost:6379')
                    await redis.ping()
                    await redis.close()
                    status = "healthy"
                    error_message = None
                except Exception as e:
                    status = "unhealthy"
                    error_message = str(e)
            
            elif service_name == 'nginx':
                # Check if nginx is responding
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        'http://localhost:80',
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status in [200, 301, 302]:
                            status = "healthy"
                            error_message = None
                        else:
                            status = "unhealthy"
                            error_message = f"HTTP {response.status}"
            
            else:
                # Docker container health check
                try:
                    container = self.docker_client.containers.get(f'cs2-{service_name}')
                    container_status = container.attrs['State']['Health']['Status']
                    
                    if container_status == 'healthy':
                        status = "healthy"
                        error_message = None
                    else:
                        status = "unhealthy"
                        error_message = f"Container status: {container_status}"
                except Exception as e:
                    status = "unhealthy"
                    error_message = str(e)
            
        except Exception as e:
            status = "unhealthy"
            error_message = str(e)
        
        response_time = time.time() - start_time
        
        return HealthCheck(
            service_name=service_name,
            status=status,
            response_time=response_time,
            last_check=datetime.now(),
            error_message=error_message
        )
    
    async def _check_all_services_health(self) -> Dict[str, HealthCheck]:
        """Check health of all services"""
        health_checks = {}
        
        tasks = []
        for service in self.config['required_services']:
            task = self._check_service_health(service)
            tasks.append((service, task))
        
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (service, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                health_checks[service] = HealthCheck(
                    service_name=service,
                    status="unhealthy",
                    response_time=0.0,
                    last_check=datetime.now(),
                    error_message=str(result)
                )
            else:
                health_checks[service] = result
        
        return health_checks
    
    async def _check_system_resources(self):
        """Check system resource availability"""
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            raise Exception(f"High memory usage: {memory.percent}%")
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            raise Exception(f"High CPU usage: {cpu_percent}%")
        
        # Load average check
        load_avg = os.getloadavg()[0]
        cpu_count = psutil.cpu_count()
        if load_avg > cpu_count * 0.8:
            raise Exception(f"High load average: {load_avg}")
        
        self.logger.info(f"✅ System resources OK - Memory: {memory.percent}%, CPU: {cpu_percent}%, Load: {load_avg}")
    
    async def _run_integration_tests(self):
        """Run integration tests after deployment"""
        self.logger.info("🧪 Running integration tests...")
        
        try:
            # Run basic API tests
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                async with session.get('http://localhost:8000/health') as response:
                    assert response.status == 200
                
                # Test API endpoints
                async with session.get('http://localhost:8000/api/status') as response:
                    assert response.status == 200
                
                # Test database connectivity
                async with session.get('http://localhost:8000/api/database/status') as response:
                    assert response.status == 200
            
            self.logger.info("✅ Integration tests passed")
            
        except Exception as e:
            raise Exception(f"Integration tests failed: {e}")
    
    async def _rollback_deployment(self, status: DeploymentStatus):
        """Rollback deployment to previous version"""
        self.logger.warning("🔄 Rolling back deployment...")
        
        if not status.rollback_version:
            raise Exception("No rollback version available")
        
        try:
            # Restore from backup
            backup_dir = Path(f"/backups/{status.rollback_version}")
            
            # Restore PostgreSQL
            postgres_backup = backup_dir / "postgres_backup.sql"
            if postgres_backup.exists():
                subprocess.run([
                    'docker-compose', 'exec', '-T', 'postgres',
                    'psql', '-U', 'cs2user', '-d', 'cs2_trading'
                ], stdin=open(postgres_backup), check=True)
            
            # Restore Redis
            redis_backup = backup_dir / "redis_backup.rdb"
            if redis_backup.exists():
                subprocess.run([
                    'docker', 'cp', str(redis_backup), 'cs2-redis:/data/dump.rdb'
                ], check=True)
                subprocess.run([
                    'docker-compose', 'restart', 'redis'
                ], check=True)
            
            # Restore application data
            app_data_backup = backup_dir / "app_data"
            if app_data_backup.exists():
                subprocess.run([
                    'cp', '-r', str(app_data_backup), './data'
                ], check=True)
            
            # Restart services
            subprocess.run(['docker-compose', 'restart'], check=True)
            
            # Update metrics
            self.metrics['rollback_count'] += 1
            
            status.status = "rollback"
            self.logger.info(f"✅ Rollback completed to version {status.rollback_version}")
            
        except Exception as e:
            self.logger.error(f"❌ Rollback failed: {e}")
            status.status = "failed"
            raise
    
    async def _simulate_deployment(self, status: DeploymentStatus):
        """Simulate deployment for dry run"""
        self.logger.info("🎭 Simulating deployment steps...")
        
        # Simulate each step with delays
        steps = [
            ("Pre-deployment checks", 5),
            ("Creating backup", 10),
            ("Pulling images", 15),
            ("Rolling update", 20),
            ("Health checks", 10),
            ("Integration tests", 8)
        ]
        
        for step_name, duration in steps:
            self.logger.info(f"🎭 Simulating: {step_name}")
            await asyncio.sleep(2)  # Reduced for testing
            self.logger.info(f"✅ Simulated: {step_name}")
        
        # Simulate health checks
        for service in self.config['required_services']:
            status.services[service] = HealthCheck(
                service_name=service,
                status="healthy",
                response_time=0.1,
                last_check=datetime.now()
            )
        
        status.status = "healthy"
        status.deployment_duration = sum(duration for _, duration in steps)
    
    def _update_deployment_metrics(self, success: bool, start_time: datetime):
        """Update deployment metrics"""
        self.metrics['deployment_count'] += 1
        
        if success:
            self.metrics['successful_deployments'] += 1
        else:
            self.metrics['failed_deployments'] += 1
        
        duration = (datetime.now() - start_time).total_seconds()
        self.metrics['average_deployment_time'] = (
            (self.metrics['average_deployment_time'] * (self.metrics['deployment_count'] - 1) + duration) /
            self.metrics['deployment_count']
        )
        
        self.metrics['last_deployment'] = datetime.now()
    
    async def _send_deployment_notifications(self, status: DeploymentStatus):
        """Send deployment notifications"""
        try:
            message = self._format_deployment_message(status)
            
            # Send to configured channels
            for channel in self.config['notification_channels']:
                if channel == 'telegram':
                    await self._send_telegram_notification(message)
                elif channel == 'slack':
                    await self._send_slack_notification(message)
            
        except Exception as e:
            self.logger.error(f"Failed to send notifications: {e}")
    
    def _format_deployment_message(self, status: DeploymentStatus) -> str:
        """Format deployment status message"""
        status_emoji = {
            "healthy": "✅",
            "failed": "❌", 
            "rollback": "🔄",
            "deploying": "🚀"
        }
        
        emoji = status_emoji.get(status.status, "❓")
        
        message = f"""
{emoji} **CS2 Float Checker Deployment**

**Environment:** {status.environment}
**Version:** {status.version}
**Status:** {status.status.upper()}
**Duration:** {status.deployment_duration:.1f}s

**Services:**
"""
        
        for service_name, health_check in status.services.items():
            service_emoji = "✅" if health_check.status == "healthy" else "❌"
            message += f"{service_emoji} {service_name}: {health_check.status}\n"
        
        if status.status == "failed" and status.rollback_version:
            message += f"\n**Rollback:** {status.rollback_version}"
        
        return message
    
    async def _send_telegram_notification(self, message: str):
        """Send Telegram notification"""
        # Implementation would use Telegram Bot API
        self.logger.info(f"📱 Sending Telegram notification: {message[:100]}...")
    
    async def _send_slack_notification(self, message: str):
        """Send Slack notification"""
        # Implementation would use Slack API
        self.logger.info(f"💬 Sending Slack notification: {message[:100]}...")
    
    async def continuous_health_monitoring(self):
        """Continuous health monitoring"""
        self.logger.info("🔍 Starting continuous health monitoring...")
        
        while True:
            try:
                health_checks = await self._check_all_services_health()
                
                unhealthy_services = [
                    name for name, check in health_checks.items() 
                    if check.status != "healthy"
                ]
                
                if unhealthy_services:
                    self.logger.warning(f"⚠️ Unhealthy services detected: {unhealthy_services}")
                    
                    # Send alert
                    alert_message = f"🚨 Health Alert: Services {unhealthy_services} are unhealthy"
                    await self._send_telegram_notification(alert_message)
                
                # Sleep for next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    def get_deployment_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics"""
        success_rate = (
            (self.metrics['successful_deployments'] / self.metrics['deployment_count'] * 100)
            if self.metrics['deployment_count'] > 0 else 0
        )
        
        return {
            **self.metrics,
            'success_rate': success_rate,
            'failure_rate': 100 - success_rate
        }

async def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CS2 Float Checker Production Deployment')
    parser.add_argument('--version', required=True, help='Version to deploy')
    parser.add_argument('--environment', default='production', help='Environment to deploy to')
    parser.add_argument('--dry-run', action='store_true', help='Simulate deployment')
    parser.add_argument('--monitor', action='store_true', help='Start health monitoring')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deployment.log'),
            logging.StreamHandler()
        ]
    )
    
    deployment = ProductionDeployment(args.environment)
    
    try:
        if args.monitor:
            await deployment.continuous_health_monitoring()
        else:
            status = await deployment.deploy(args.version, dry_run=args.dry_run)
            print(f"\n🎯 Deployment completed with status: {status.status}")
            
            # Print metrics
            metrics = deployment.get_deployment_metrics()
            print(f"📊 Deployment Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
    
    except KeyboardInterrupt:
        print("\n👋 Deployment interrupted by user")
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())