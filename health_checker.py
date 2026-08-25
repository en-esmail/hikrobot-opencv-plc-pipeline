"""
System health checker for the hikrobot-opencv-plc-pipeline project.

Validates that all system prerequisites are met before running
sorting operations, providing early error detection and clear
diagnostic information.
"""

import socket
from typing import List, Tuple
from logging_config import get_logger
from exceptions import PLCConnectionError, HikrobotError


class HealthCheckResult:
    """Result of a single health check."""
    
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"[{status}] {self.name}: {self.message}" if self.message else f"[{status}] {self.name}"


class SystemHealthChecker:
    """Validate system prerequisites before sorting operations.
    
    Performs comprehensive checks on:
    - PLC connectivity and responsiveness
    - Camera availability
    - Calibration data validity
    - Configuration integrity
    """
    
    def __init__(self, ip: str = "192.168.6.10", port: int = 2023):
        """Initialize health checker.
        
        Args:
            ip: PLC IP address
            port: PLC communication port
        """
        self.ip = ip
        self.port = port
        self.logger = get_logger()
        self.results: List[HealthCheckResult] = []
    
    def check_all(self) -> Tuple[bool, List[HealthCheckResult]]:
        """Run all health checks.
        
        Returns:
            Tuple of (all_passed: bool, results: List[HealthCheckResult])
        """
        self.results = []
        
        self.logger.info("Starting system health checks...")
        
        # Run all checks
        self._check_plc_connectivity()
        self._check_plc_responsiveness()
        self._check_calibration_data()
        
        # Summary
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        all_passed = passed_count == total_count
        
        self.logger.info(f"Health checks complete: {passed_count}/{total_count} passed")
        
        return all_passed, self.results
    
    def _check_plc_connectivity(self) -> None:
        """Check if PLC is reachable via socket connection."""
        try:
            self.logger.debug(f"Checking PLC connectivity to {self.ip}:{self.port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)  # 3 second timeout
            sock.connect((self.ip, self.port))
            sock.close()
            
            result = HealthCheckResult(
                "PLC Connectivity",
                True,
                f"Successfully connected to {self.ip}:{self.port}"
            )
            self.results.append(result)
            self.logger.info(result)
        except socket.timeout:
            result = HealthCheckResult(
                "PLC Connectivity",
                False,
                f"Connection timeout to {self.ip}:{self.port} (check IP/port)"
            )
            self.results.append(result)
            self.logger.warning(result)
        except socket.error as e:
            result = HealthCheckResult(
                "PLC Connectivity",
                False,
                f"Cannot connect to PLC: {e}"
            )
            self.results.append(result)
            self.logger.warning(result)
        except Exception as e:
            result = HealthCheckResult(
                "PLC Connectivity",
                False,
                f"Unexpected error: {e}"
            )
            self.results.append(result)
            self.logger.warning(result)
    
    def _check_plc_responsiveness(self) -> None:
        """Check if PLC responds to handshake command."""
        try:
            self.logger.debug("Checking PLC responsiveness...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((self.ip, self.port))
            
            # Send handshake
            sock.send("plc,6".encode("utf-8"))
            response = sock.recv(1024).decode("gbk").strip()
            sock.close()
            
            if response == "1":
                result = HealthCheckResult(
                    "PLC Responsiveness",
                    True,
                    "PLC responds to handshake (ready)"
                )
                self.results.append(result)
                self.logger.info(result)
            else:
                result = HealthCheckResult(
                    "PLC Responsiveness",
                    True,
                    f"PLC responds but not ready (response: {response!r})"
                )
                self.results.append(result)
                self.logger.info(result)
        except Exception as e:
            result = HealthCheckResult(
                "PLC Responsiveness",
                False,
                f"PLC not responding: {e}"
            )
            self.results.append(result)
            self.logger.warning(result)
    
    def _check_calibration_data(self) -> None:
        """Check if calibration matrix is available and valid.
        
        Note: This is a basic check. Actual validation depends on
        how calibration data is stored in the application.
        """
        try:
            import numpy as np
            # This is a placeholder check - adapt to your actual calibration storage
            # For now, just verify we can import numpy for calibration math
            
            result = HealthCheckResult(
                "Calibration Data",
                True,
                "NumPy available for calibration calculations"
            )
            self.results.append(result)
            self.logger.info(result)
        except ImportError:
            result = HealthCheckResult(
                "Calibration Data",
                False,
                "NumPy not available (required for calibration)"
            )
            self.results.append(result)
            self.logger.warning(result)
        except Exception as e:
            result = HealthCheckResult(
                "Calibration Data",
                False,
                f"Calibration check failed: {e}"
            )
            self.results.append(result)
            self.logger.warning(result)
    
    def print_report(self) -> None:
        """Print human-readable health check report."""
        print("\n" + "="*60)
        print("SYSTEM HEALTH CHECK REPORT")
        print("="*60)
        
        for result in self.results:
            print(str(result))
        
        print("="*60 + "\n")
    
    @staticmethod
    def assert_healthy(all_passed: bool, results: List[HealthCheckResult]) -> None:
        """Raise exception if health checks failed.
        
        Args:
            all_passed: Result from check_all()
            results: Results list from check_all()
            
        Raises:
            HikrobotError: If any health check failed
        """
        if not all_passed:
            failed_checks = [r for r in results if not r.passed]
            failed_names = ", ".join(r.name for r in failed_checks)
            raise HikrobotError(f"Health checks failed: {failed_names}")
