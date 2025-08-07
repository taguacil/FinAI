"""
Health check system for Portfolio Tracker.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..data_providers.manager import DataProviderManager
from ..portfolio.storage import FileBasedStorage
from .logging_config import get_logger


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    service: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    details: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class HealthChecker:
    """Comprehensive health checker for all system components."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.logger = get_logger(__name__)
        self.storage = FileBasedStorage(data_dir)
        self.data_manager = DataProviderManager()

    def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks and return results."""
        self.logger.info("Starting comprehensive health check")

        checks = {
            "storage": self.check_storage_health,
            "data_providers": self.check_data_providers_health,
            "portfolio_data": self.check_portfolio_data_health,
            "system_resources": self.check_system_resources,
            "dependencies": self.check_dependencies,
        }

        results = {}
        for check_name, check_func in checks.items():
            try:
                start_time = time.time()
                result = check_func()
                result.response_time_ms = (time.time() - start_time) * 1000
                results[check_name] = result

                self.logger.debug(
                    f"Health check '{check_name}': {result.status} - {result.message}"
                )
            except Exception as e:
                results[check_name] = HealthCheckResult(
                    service=check_name,
                    status="critical",
                    message=f"Health check failed: {e}",
                    response_time_ms=(
                        (time.time() - start_time) * 1000
                        if "start_time" in locals()
                        else None
                    ),
                )
                self.logger.error(f"Health check '{check_name}' failed: {e}")

        overall_status = self._determine_overall_status(results)
        self.logger.info(f"Health check completed - Overall status: {overall_status}")

        return results

    def check_storage_health(self) -> HealthCheckResult:
        """Check storage system health."""
        try:
            # Check if data directory exists and is writable
            if not self.data_dir.exists():
                return HealthCheckResult(
                    service="storage",
                    status="critical",
                    message="Data directory does not exist",
                )

            # Test write permission
            test_file = self.data_dir / ".health_check"
            test_file.write_text("health check")
            test_file.unlink()

            # Check subdirectories
            subdirs = ["portfolios", "snapshots", "logs", "exports", "backups"]
            missing_dirs = []
            for subdir in subdirs:
                if not (self.data_dir / subdir).exists():
                    missing_dirs.append(subdir)

            # Check disk space (basic check)
            import shutil

            total, used, free = shutil.disk_usage(self.data_dir)
            free_gb = free // (1024**3)

            details = {
                "data_directory": str(self.data_dir),
                "writable": True,
                "missing_subdirs": missing_dirs,
                "free_space_gb": free_gb,
                "total_space_gb": total // (1024**3),
                "used_space_gb": used // (1024**3),
            }

            if free_gb < 1:
                return HealthCheckResult(
                    service="storage",
                    status="warning",
                    message=f"Low disk space: {free_gb}GB free",
                    details=details,
                )
            elif missing_dirs:
                return HealthCheckResult(
                    service="storage",
                    status="warning",
                    message=f"Missing subdirectories: {', '.join(missing_dirs)}",
                    details=details,
                )
            else:
                return HealthCheckResult(
                    service="storage",
                    status="healthy",
                    message=f"Storage system operational ({free_gb}GB free)",
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                service="storage",
                status="critical",
                message=f"Storage system failure: {e}",
            )

    def check_data_providers_health(self) -> HealthCheckResult:
        """Check data provider connectivity and functionality."""
        try:
            provider_results = {}

            # Test each provider with a simple request
            for provider in self.data_manager.providers:
                provider_name = provider.name
                try:
                    # Test with a well-known symbol
                    start_time = time.time()
                    price = provider.get_current_price("AAPL")
                    response_time = (time.time() - start_time) * 1000

                    if price is not None:
                        provider_results[provider_name] = {
                            "status": "healthy",
                            "response_time_ms": response_time,
                            "test_price": float(price),
                        }
                    else:
                        provider_results[provider_name] = {
                            "status": "warning",
                            "response_time_ms": response_time,
                            "error": "No price data returned",
                        }

                except Exception as e:
                    provider_results[provider_name] = {
                        "status": "critical",
                        "error": str(e),
                    }

            # Determine overall status
            healthy_providers = [
                p for p in provider_results.values() if p["status"] == "healthy"
            ]
            total_providers = len(provider_results)

            if len(healthy_providers) == total_providers:
                status = "healthy"
                message = f"All {total_providers} data providers operational"
            elif len(healthy_providers) > 0:
                status = "warning"
                message = f"{len(healthy_providers)}/{total_providers} data providers operational"
            else:
                status = "critical"
                message = "No data providers operational"

            return HealthCheckResult(
                service="data_providers",
                status=status,
                message=message,
                details=provider_results,
            )

        except Exception as e:
            return HealthCheckResult(
                service="data_providers",
                status="critical",
                message=f"Data provider check failed: {e}",
            )

    def check_portfolio_data_health(self) -> HealthCheckResult:
        """Check portfolio data integrity and accessibility."""
        try:
            # Get all portfolios
            portfolio_ids = self.storage.list_portfolios()

            if not portfolio_ids:
                return HealthCheckResult(
                    service="portfolio_data",
                    status="warning",
                    message="No portfolios found",
                    details={"portfolio_count": 0},
                )

            portfolio_health = []
            corrupted_portfolios = []

            for portfolio_id in portfolio_ids[:10]:  # Check up to 10 portfolios
                try:
                    portfolio = self.storage.load_portfolio(portfolio_id)
                    if portfolio:
                        # Check portfolio integrity
                        snapshots = self.storage.load_snapshots(portfolio_id)

                        portfolio_health.append(
                            {
                                "id": portfolio_id,
                                "name": portfolio.name,
                                "transactions": len(portfolio.transactions),
                                "positions": len(portfolio.positions),
                                "snapshots": len(snapshots),
                                "status": "healthy",
                            }
                        )
                    else:
                        corrupted_portfolios.append(portfolio_id)

                except Exception as e:
                    corrupted_portfolios.append(f"{portfolio_id} ({e})")

            details = {
                "total_portfolios": len(portfolio_ids),
                "checked_portfolios": len(portfolio_health),
                "healthy_portfolios": len(portfolio_health),
                "corrupted_portfolios": corrupted_portfolios,
                "portfolio_sample": portfolio_health[:5],  # First 5 for details
            }

            if corrupted_portfolios:
                return HealthCheckResult(
                    service="portfolio_data",
                    status="warning",
                    message=f"{len(corrupted_portfolios)} portfolios have issues",
                    details=details,
                )
            else:
                return HealthCheckResult(
                    service="portfolio_data",
                    status="healthy",
                    message=f"All {len(portfolio_health)} checked portfolios are healthy",
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                service="portfolio_data",
                status="critical",
                message=f"Portfolio data check failed: {e}",
            )

    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available // (1024**3)

            # Disk usage for data directory
            disk_usage = psutil.disk_usage(str(self.data_dir))
            disk_free_gb = disk_usage.free // (1024**3)
            disk_percent = (disk_usage.used / disk_usage.total) * 100

            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_gb": memory_available_gb,
                "disk_free_gb": disk_free_gb,
                "disk_used_percent": disk_percent,
            }

            # Determine status based on thresholds
            issues = []
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            if memory_percent > 90:
                issues.append(f"High memory usage: {memory_percent}%")
            if disk_free_gb < 1:
                issues.append(f"Low disk space: {disk_free_gb}GB free")

            if issues:
                return HealthCheckResult(
                    service="system_resources",
                    status="warning",
                    message="; ".join(issues),
                    details=details,
                )
            else:
                return HealthCheckResult(
                    service="system_resources",
                    status="healthy",
                    message="System resources within normal limits",
                    details=details,
                )

        except ImportError:
            return HealthCheckResult(
                service="system_resources",
                status="unknown",
                message="psutil not available for resource monitoring",
            )
        except Exception as e:
            return HealthCheckResult(
                service="system_resources",
                status="warning",
                message=f"Could not check system resources: {e}",
            )

    def check_dependencies(self) -> HealthCheckResult:
        """Check critical dependencies."""
        try:
            dependencies = {
                "pydantic": "Core data models",
                "pandas": "Data manipulation",
                "numpy": "Numerical calculations",
                "yfinance": "Yahoo Finance data",
                "requests": "HTTP requests",
                "streamlit": "Web UI",
            }

            missing_deps = []
            available_deps = {}

            for dep_name, description in dependencies.items():
                try:
                    module = __import__(dep_name)
                    version = getattr(module, "__version__", "unknown")
                    available_deps[dep_name] = {
                        "version": version,
                        "description": description,
                        "status": "available",
                    }
                except ImportError:
                    missing_deps.append(dep_name)
                    available_deps[dep_name] = {
                        "status": "missing",
                        "description": description,
                    }

            details = {
                "total_dependencies": len(dependencies),
                "available_dependencies": len(dependencies) - len(missing_deps),
                "missing_dependencies": missing_deps,
                "dependency_details": available_deps,
            }

            if missing_deps:
                return HealthCheckResult(
                    service="dependencies",
                    status="critical",
                    message=f"Missing critical dependencies: {', '.join(missing_deps)}",
                    details=details,
                )
            else:
                return HealthCheckResult(
                    service="dependencies",
                    status="healthy",
                    message=f"All {len(dependencies)} dependencies available",
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                service="dependencies",
                status="warning",
                message=f"Dependency check failed: {e}",
            )

    def _determine_overall_status(self, results: Dict[str, HealthCheckResult]) -> str:
        """Determine overall system health from individual checks."""
        statuses = [result.status for result in results.values()]

        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"

    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health."""
        results = self.check_all()

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self._determine_overall_status(results),
            "services": {
                name: {
                    "status": result.status,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                }
                for name, result in results.items()
            },
            "summary": {
                "total_services": len(results),
                "healthy_services": len(
                    [r for r in results.values() if r.status == "healthy"]
                ),
                "warning_services": len(
                    [r for r in results.values() if r.status == "warning"]
                ),
                "critical_services": len(
                    [r for r in results.values() if r.status == "critical"]
                ),
            },
        }
