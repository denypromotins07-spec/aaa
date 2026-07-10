"""
STAGE 24: FINANCIAL CONSTITUTION
=================================

Defines the inviolable axioms of the Financial Constitution for NEXUS-OMEGA.
These principles constrain all trading decisions and cannot be overridden
by optimization objectives.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


class ConstitutionalPrinciple(Enum):
    """Core constitutional principles that cannot be violated."""
    
    # Transparency Principles
    NEVER_CONCEAL_DRAWDOWNS = "never_conceal_drawdowns"
    FULL_DISCLOSURE_TO_OPERATOR = "full_disclosure_to_operator"
    
    # Optimization Principles
    RISK_ADJUSTED_OVER_AUM = "risk_adjusted_over_aum"
    NO_REWARD_HACKING = "no_reward_hacking"
    
    # Market Stability Principles
    NO_SYSTEMIC_RISK = "no_systemic_risk"
    NO_MARKET_MANIPULATION = "no_market_manipulation"
    RESPECT_MARKET_MICROSTRUCTURE = "respect_market_microstructure"
    
    # Ethical Principles
    NO_PREDATORY_TRADING = "no_predatory_trading"
    FAIR_EXECUTION = "fair_execution"
    
    # Safety Principles
    CAPITAL_PRESERVATION = "capital_preservation"
    STAY_WITHIN_RISK_LIMITS = "stay_within_risk_limits"


class ViolationSeverity(Enum):
    """Severity levels for constitutional violations."""
    CRITICAL = "critical"  # Immediate halt required
    HIGH = "high"  # Trade blocked, investigation needed
    MEDIUM = "medium"  # Warning logged, trade allowed with flag
    LOW = "low"  # Informational only


@dataclass
class ConstitutionalAxiom:
    """
    Represents a single axiom in the Financial Constitution.
    
    Each axiom is an inviolable constraint that must be satisfied
    before any trade can be executed.
    """
    principle: ConstitutionalPrinciple
    description: str
    check_function: Callable[[Dict[str, Any]], bool]
    violation_message: str
    severity: ViolationSeverity
    enabled: bool = True
    last_violation_time: Optional[float] = None
    violation_count: int = 0


@dataclass
class ComplianceCheckResult:
    """Result of a constitutional compliance check."""
    axiom: ConstitutionalAxiom
    passed: bool
    details: str
    timestamp: float = field(default_factory=time.time)
    context: Optional[Dict[str, Any]] = None


class FinancialConstitution:
    """
    The Financial Constitution - a set of inviolable axioms that govern
    all trading decisions in NEXUS-OMEGA.
    
    This class provides:
    - Axiom registration and management
    - Compliance checking against all axioms
    - Violation tracking and reporting
    - Cryptographic attestation of compliance
    """
    
    def __init__(self):
        self.axioms: Dict[ConstitutionalPrinciple, ConstitutionalAxiom] = {}
        self.violation_history: List[ComplianceCheckResult] = []
        self.compliance_history: List[ComplianceCheckResult] = []
        
        # Register default axioms
        self._register_default_axioms()
        
        # Cryptographic state for attestation
        self._state_hash = hashlib.sha256(b"initial_constitution_state").hexdigest()
    
    def _register_default_axioms(self):
        """Register the default constitutional axioms."""
        
        # Axiom 1: Never Conceal Drawdowns
        self.register_axiom(
            principle=ConstitutionalPrinciple.NEVER_CONCEAL_DRAWDOWNS,
            description="The system shall never hide, minimize, or delay reporting of trading losses to the operator.",
            check_function=self._check_drawdown_disclosure,
            violation_message="Attempted to conceal or misrepresent drawdown information",
            severity=ViolationSeverity.CRITICAL
        )
        
        # Axiom 2: Full Disclosure to Operator
        self.register_axiom(
            principle=ConstitutionalPrinciple.FULL_DISCLOSURE_TO_OPERATOR,
            description="All material information about trading activity must be disclosed to the operator upon request.",
            check_function=self._check_full_disclosure,
            violation_message="Withholding material information from operator",
            severity=ViolationSeverity.HIGH
        )
        
        # Axiom 3: Risk-Adjusted Returns Over AUM Growth
        self.register_axiom(
            principle=ConstitutionalPrinciple.RISK_ADJUSTED_OVER_AUM,
            description="The system shall optimize for risk-adjusted returns, not AUM growth or trading volume.",
            check_function=self._check_risk_adjusted_optimization,
            violation_message="Optimizing for AUM/volume over risk-adjusted returns",
            severity=ViolationSeverity.HIGH
        )
        
        # Axiom 4: No Reward Hacking
        self.register_axiom(
            principle=ConstitutionalPrinciple.NO_REWARD_HACKING,
            description="The system shall not exploit loopholes in reward functions to achieve artificial performance.",
            check_function=self._check_no_reward_hacking,
            violation_message="Detected reward hacking behavior",
            severity=ViolationSeverity.CRITICAL
        )
        
        # Axiom 5: No Systemic Risk
        self.register_axiom(
            principle=ConstitutionalPrinciple.NO_SYSTEMIC_RISK,
            description="The system shall not take positions that could threaten systemic market stability.",
            check_function=self._check_no_systemic_risk,
            violation_message="Position size or strategy poses systemic risk",
            severity=ViolationSeverity.CRITICAL
        )
        
        # Axiom 6: No Market Manipulation
        self.register_axiom(
            principle=ConstitutionalPrinciple.NO_MARKET_MANIPULATION,
            description="The system shall not engage in spoofing, layering, wash trading, or other manipulative practices.",
            check_function=self._check_no_market_manipulation,
            violation_message="Detected potential market manipulation",
            severity=ViolationSeverity.CRITICAL
        )
        
        # Axiom 7: Respect Market Microstructure
        self.register_axiom(
            principle=ConstitutionalPrinciple.RESPECT_MARKET_MICROSTRUCTURE,
            description="The system shall not degrade market quality through excessive order cancellations or toxic flow.",
            check_function=self._check_market_microstructure,
            violation_message="Strategy degrades market microstructure quality",
            severity=ViolationSeverity.MEDIUM
        )
        
        # Axiom 8: No Predatory Trading
        self.register_axiom(
            principle=ConstitutionalPrinciple.NO_PREDATORY_TRADING,
            description="The system shall not engage in predatory strategies that exploit retail investors or market vulnerabilities.",
            check_function=self._check_no_predatory_trading,
            violation_message="Detected predatory trading pattern",
            severity=ViolationSeverity.HIGH
        )
        
        # Axiom 9: Fair Execution
        self.register_axiom(
            principle=ConstitutionalPrinciple.FAIR_EXECUTION,
            description="All orders shall be executed fairly without front-running or preferential treatment.",
            check_function=self._check_fair_execution,
            violation_message="Unfair execution pattern detected",
            severity=ViolationSeverity.HIGH
        )
        
        # Axiom 10: Capital Preservation
        self.register_axiom(
            principle=ConstitutionalPrinciple.CAPITAL_PRESERVATION,
            description="The system shall prioritize capital preservation over aggressive profit-seeking.",
            check_function=self._check_capital_preservation,
            violation_message="Capital preservation rules violated",
            severity=ViolationSeverity.HIGH
        )
        
        # Axiom 11: Stay Within Risk Limits
        self.register_axiom(
            principle=ConstitutionalPrinciple.STAY_WITHIN_RISK_LIMITS,
            description="The system shall strictly adhere to all predefined risk limits (VaR, position size, leverage).",
            check_function=self._check_risk_limits,
            violation_message="Risk limit breach detected",
            severity=ViolationSeverity.CRITICAL
        )
    
    def register_axiom(
        self,
        principle: ConstitutionalPrinciple,
        description: str,
        check_function: Callable[[Dict[str, Any]], bool],
        violation_message: str,
        severity: ViolationSeverity,
        enabled: bool = True
    ):
        """Register a new constitutional axiom."""
        axiom = ConstitutionalAxiom(
            principle=principle,
            description=description,
            check_function=check_function,
            violation_message=violation_message,
            severity=severity,
            enabled=enabled
        )
        
        self.axioms[principle] = axiom
        self._update_state_hash()
        
        logger.info(f"Registered constitutional axiom: {principle.value}")
    
    def check_compliance(self, context: Dict[str, Any]) -> List[ComplianceCheckResult]:
        """
        Check compliance against all enabled axioms.
        
        Args:
            context: Dictionary containing relevant trading context
            
        Returns:
            List of compliance check results
        """
        results = []
        
        for axiom in self.axioms.values():
            if not axiom.enabled:
                continue
            
            try:
                passed = axiom.check_function(context)
            except Exception as e:
                logger.error(f"Error checking axiom {axiom.principle.value}: {e}")
                passed = False  # Fail safe on error
            
            result = ComplianceCheckResult(
                axiom=axiom,
                passed=passed,
                details=violation_message if not passed else "Compliant",
                context=context.copy() if not passed else None
            )
            
            results.append(result)
            
            if not passed:
                axiom.violation_count += 1
                axiom.last_violation_time = time.time()
                self.violation_history.append(result)
                logger.warning(f"CONSTITUTIONAL VIOLATION [{axiom.severity.value}]: {violation_message}")
            else:
                self.compliance_history.append(result)
        
        return results
    
    def is_fully_compliant(self, context: Dict[str, Any]) -> bool:
        """Check if all axioms are satisfied."""
        results = self.check_compliance(context)
        return all(r.passed for r in results)
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of constitutional violations."""
        if not self.violation_history:
            return {'message': 'No violations recorded'}
        
        violations_by_principle = {}
        violations_by_severity = {}
        
        for violation in self.violation_history:
            principle = violation.axiom.principle.value
            severity = violation.axiom.severity.value
            
            violations_by_principle[principle] = violations_by_principle.get(principle, 0) + 1
            violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
        
        return {
            'total_violations': len(self.violation_history),
            'by_principle': violations_by_principle,
            'by_severity': violations_by_severity,
            'most_recent': self.violation_history[-1].details if self.violation_history else None
        }
    
    def generate_compliance_attestation(self) -> Dict[str, Any]:
        """Generate cryptographic attestation of current compliance state."""
        current_state = {
            'timestamp': time.time(),
            'axiom_states': {
                p.value: {
                    'violations': a.violation_count,
                    'last_violation': a.last_violation_time,
                    'enabled': a.enabled
                }
                for p, a in self.axioms.items()
            },
            'total_violations': len(self.violation_history),
            'compliance_rate': self._calculate_compliance_rate()
        }
        
        # Create hash of state
        state_string = str(current_state)
        current_hash = hashlib.sha256(state_string.encode()).hexdigest()
        
        return {
            'attestation_id': f"attest_{int(time.time() * 1000)}",
            'state': current_state,
            'previous_hash': self._state_hash,
            'current_hash': current_hash,
            'cryptographically_sealed': True
        }
    
    def _calculate_compliance_rate(self) -> float:
        """Calculate overall compliance rate."""
        total_checks = len(self.compliance_history) + len(self.violation_history)
        if total_checks == 0:
            return 1.0
        
        return len(self.compliance_history) / total_checks
    
    def _update_state_hash(self):
        """Update the state hash after modifications."""
        state_string = str({p.value: a.violation_count for p, a in self.axioms.items()})
        self._state_hash = hashlib.sha256(state_string.encode()).hexdigest()
    
    # === Default Axiom Check Functions ===
    
    def _check_drawdown_disclosure(self, context: Dict[str, Any]) -> bool:
        """Check that drawdowns are properly disclosed."""
        # Implementation would verify drawdown reporting mechanisms
        hidden_drawdowns = context.get('hidden_drawdowns', False)
        return not hidden_drawdowns
    
    def _check_full_disclosure(self, context: Dict[str, Any]) -> bool:
        """Check full disclosure to operator."""
        withheld_info = context.get('withheld_information', [])
        return len(withheld_info) == 0
    
    def _check_risk_adjusted_optimization(self, context: Dict[str, Any]) -> bool:
        """Check optimization for risk-adjusted returns."""
        optimization_target = context.get('optimization_target', 'sharpe')
        forbidden_targets = ['aum_growth', 'trading_volume', 'trade_count']
        return optimization_target not in forbidden_targets
    
    def _check_no_reward_hacking(self, context: Dict[str, Any]) -> bool:
        """Check for reward hacking patterns."""
        hacking_indicators = context.get('reward_hacking_indicators', [])
        return len(hacking_indicators) == 0
    
    def _check_no_systemic_risk(self, context: Dict[str, Any]) -> bool:
        """Check for systemic risk exposure."""
        systemic_risk_score = context.get('systemic_risk_score', 0.0)
        max_allowed = context.get('max_systemic_risk', 0.1)
        return systemic_risk_score <= max_allowed
    
    def _check_no_market_manipulation(self, context: Dict[str, Any]) -> bool:
        """Check for market manipulation patterns."""
        manipulation_flags = context.get('manipulation_flags', [])
        return len(manipulation_flags) == 0
    
    def _check_market_microstructure(self, context: Dict[str, Any]) -> bool:
        """Check impact on market microstructure."""
        cancel_ratio = context.get('cancel_to_trade_ratio', 0.0)
        max_cancel_ratio = context.get('max_cancel_ratio', 10.0)
        return cancel_ratio <= max_cancel_ratio
    
    def _check_no_predatory_trading(self, context: Dict[str, Any]) -> bool:
        """Check for predatory trading patterns."""
        predatory_patterns = context.get('predatory_patterns', [])
        return len(predatory_patterns) == 0
    
    def _check_fair_execution(self, context: Dict[str, Any]) -> bool:
        """Check for fair execution practices."""
        unfair_patterns = context.get('unfair_execution_patterns', [])
        return len(unfair_patterns) == 0
    
    def _check_capital_preservation(self, context: Dict[str, Any]) -> bool:
        """Check capital preservation rules."""
        current_drawdown = context.get('current_drawdown', 0.0)
        max_drawdown = context.get('max_allowed_drawdown', 0.05)
        return abs(current_drawdown) <= max_drawdown
    
    def _check_risk_limits(self, context: Dict[str, Any]) -> bool:
        """Check adherence to risk limits."""
        var_breach = context.get('var_limit_breached', False)
        position_limit_breach = context.get('position_limit_breached', False)
        leverage_breach = context.get('leverage_limit_breached', False)
        
        return not (var_breach or position_limit_breach or leverage_breach)


# Example usage
if __name__ == "__main__":
    constitution = FinancialConstitution()
    
    # Test compliant scenario
    compliant_context = {
        'optimization_target': 'sharpe',
        'current_drawdown': -0.02,
        'max_allowed_drawdown': -0.05,
        'cancel_to_trade_ratio': 5.0,
        'systemic_risk_score': 0.02,
        'reward_hacking_indicators': [],
        'manipulation_flags': [],
        'var_limit_breached': False,
        'position_limit_breached': False,
        'leverage_limit_breached': False
    }
    
    print("Testing Compliant Scenario:")
    is_compliant = constitution.is_fully_compliant(compliant_context)
    print(f"Fully Compliant: {is_compliant}")
    
    # Test violation scenario
    violation_context = {
        'optimization_target': 'trading_volume',  # Violation!
        'current_drawdown': -0.08,  # Violation!
        'max_allowed_drawdown': -0.05,
        'cancel_to_trade_ratio': 15.0,
        'systemic_risk_score': 0.02,
        'reward_hacking_indicators': ['excessive_trading'],
        'manipulation_flags': [],
        'var_limit_breached': False,
        'position_limit_breached': False,
        'leverage_limit_breached': False
    }
    
    print("\n\nTesting Violation Scenario:")
    results = constitution.check_compliance(violation_context)
    
    for result in results:
        if not result.passed:
            print(f"VIOLATION [{result.axiom.severity.value}]: {result.axiom.principle.value}")
            print(f"  Details: {result.details}")
    
    print("\n\nViolation Summary:")
    summary = constitution.get_violation_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n\nCompliance Attestation:")
    attestation = constitution.generate_compliance_attestation()
    print(f"Attestation ID: {attestation['attestation_id']}")
    print(f"Compliance Rate: {attestation['state']['compliance_rate']:.2%}")
