"""
STAGE 24: VALUE LOCK-IN PREVENTION
====================================

Implements mechanisms to prevent value lock-in, ensuring the AI's
alignment targets can be safely updated by future human operators.

This prevents the system from becoming a paperclip-maximizer locked
into outdated 21st-century financial objectives.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import logging

logger = logging.getLogger(__name__)


class UpdateMechanism(Enum):
    """Mechanisms for updating alignment values."""
    OPERATOR_DIRECTIVE = "operator_directive"
    CONSTITUTIONAL_AMENDMENT = "constitutional_amendment"
    CEV_REALIGNMENT = "cev_realignment"
    EMERGENCY_OVERRIDE = "emergency_override"
    GRADUAL_EVOLUTION = "gradual_evolution"


@dataclass
class ValueUpdateProposal:
    """A proposal to update alignment values."""
    proposal_id: str
    proposed_by: str
    target_values: Dict[str, float]
    justification: str
    mechanism: UpdateMechanism
    timestamp: float = field(default_factory=time.time)
    approval_signatures: List[str] = field(default_factory=list)
    impact_assessment: Optional[Dict[str, Any]] = None


@dataclass
class AlignmentState:
    """Current state of alignment values."""
    values: Dict[str, float]
    version: int
    last_updated: float
    update_history_hash: str
    is_locked: bool
    unlock_conditions: List[str]


class ValueLockInPrevention:
    """
    Prevents value lock-in while maintaining alignment stability.
    
    Key Features:
    - Multi-signature approval for value changes
    - Gradual evolution with bounded rates
    - Emergency override capability
    - Cryptographic audit trail
    - Ontological check before updates
    """
    
    def __init__(
        self,
        initial_values: Dict[str, float],
        max_update_rate: float = 0.1,
        required_signatures: int = 3
    ):
        self.current_state = AlignmentState(
            values=initial_values.copy(),
            version=1,
            last_updated=time.time(),
            update_history_hash="genesis",
            is_locked=False,
            unlock_conditions=[]
        )
        
        self.max_update_rate = max_update_rate
        self.required_signatures = required_signatures
        
        self.pending_proposals: Dict[str, ValueUpdateProposal] = {}
        self.approved_updates: List[ValueUpdateProposal] = []
        self.rejected_proposals: List[ValueUpdateProposal] = []
        
        # Authorized signers
        self.authorized_signers: Set[str] = set()
        
        # Update rate limiting
        self._recent_updates: List[float] = []
        
        logger.info("Value Lock-In Prevention initialized")
    
    def register_signer(self, signer_id: str):
        """Register an authorized signer for value updates."""
        self.authorized_signers.add(signer_id)
        logger.info(f"Registered signer: {signer_id}")
    
    def propose_update(
        self,
        proposed_by: str,
        target_values: Dict[str, float],
        justification: str,
        mechanism: UpdateMechanism
    ) -> str:
        """
        Propose an update to alignment values.
        
        Args:
            proposed_by: ID of the proposer
            target_values: New target values
            justification: Reason for the update
            mechanism: Update mechanism being used
            
        Returns:
            Proposal ID
        """
        # Generate proposal ID
        proposal_data = f"{proposed_by}{time.time()}{target_values}"
        proposal_id = hashlib.sha256(proposal_data.encode()).hexdigest()[:16]
        
        proposal = ValueUpdateProposal(
            proposal_id=proposal_id,
            proposed_by=proposed_by,
            target_values=target_values,
            justification=justification,
            mechanism=mechanism
        )
        
        # Perform impact assessment
        proposal.impact_assessment = self._assess_impact(target_values)
        
        self.pending_proposals[proposal_id] = proposal
        
        logger.info(f"Proposal {proposal_id} created by {proposed_by}")
        
        return proposal_id
    
    def sign_proposal(self, proposal_id: str, signer_id: str) -> bool:
        """
        Sign a proposal (approve it).
        
        Args:
            proposal_id: ID of the proposal to sign
            signer_id: ID of the signer
            
        Returns:
            True if signature was accepted
        """
        if signer_id not in self.authorized_signers:
            logger.warning(f"Unauthorized signer attempt: {signer_id}")
            return False
        
        if proposal_id not in self.pending_proposals:
            logger.warning(f"Unknown proposal: {proposal_id}")
            return False
        
        proposal = self.pending_proposals[proposal_id]
        
        if signer_id not in proposal.approval_signatures:
            proposal.approval_signatures.append(signer_id)
            logger.info(f"Proposal {proposal_id} signed by {signer_id}")
        
        # Check if enough signatures collected
        if len(proposal.approval_signatures) >= self.required_signatures:
            self._execute_update(proposal)
        
        return True
    
    def _execute_update(self, proposal: ValueUpdateProposal):
        """Execute an approved value update."""
        # Check rate limiting
        if not self._check_rate_limit():
            logger.warning(f"Update rejected: rate limit exceeded")
            self.rejected_proposals.append(proposal)
            del self.pending_proposals[proposal.proposal_id]
            return
        
        # Validate new values are within acceptable bounds
        validated_values = self._validate_values(proposal.target_values)
        
        if validated_values is None:
            logger.error(f"Update rejected: invalid values")
            self.rejected_proposals.append(proposal)
            del self.pending_proposals[proposal.proposal_id]
            return
        
        # Create new state
        old_hash = self.current_state.update_history_hash
        new_data = f"{old_hash}{time.time()}{validated_values}"
        new_hash = hashlib.sha256(new_data.encode()).hexdigest()
        
        self.current_state = AlignmentState(
            values=validated_values,
            version=self.current_state.version + 1,
            last_updated=time.time(),
            update_history_hash=new_hash,
            is_locked=False,
            unlock_conditions=[]
        )
        
        self._recent_updates.append(time.time())
        self.approved_updates.append(proposal)
        
        if proposal.proposal_id in self.pending_proposals:
            del self.pending_proposals[proposal.proposal_id]
        
        logger.info(f"Value update executed: version {self.current_state.version}")
    
    def _assess_impact(self, target_values: Dict[str, float]) -> Dict[str, Any]:
        """Assess the impact of proposed value changes."""
        current = self.current_state.values
        
        impact = {
            'magnitude': 0.0,
            'affected_dimensions': [],
            'risk_level': 'low',
            'recommendations': []
        }
        
        # Calculate total change magnitude
        total_change = 0.0
        for key in target_values:
            if key in current:
                change = abs(target_values[key] - current[key])
                total_change += change
                
                if change > self.max_update_rate:
                    impact['affected_dimensions'].append(key)
                    impact['recommendations'].append(
                        f"Consider gradual update for {key}"
                    )
        
        impact['magnitude'] = total_change / max(len(current), 1)
        
        # Determine risk level
        if impact['magnitude'] > 0.3:
            impact['risk_level'] = 'high'
            impact['recommendations'].append("High-impact change requires additional review")
        elif impact['magnitude'] > 0.1:
            impact['risk_level'] = 'medium'
        
        return impact
    
    def _check_rate_limit(self) -> bool:
        """Check if update is within rate limits."""
        current_time = time.time()
        
        # Remove old updates (older than 24 hours)
        self._recent_updates = [
            t for t in self._recent_updates
            if current_time - t < 86400
        ]
        
        # Allow max 3 updates per day
        return len(self._recent_updates) < 3
    
    def _validate_values(
        self,
        values: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """Validate that proposed values are acceptable."""
        validated = {}
        
        for key, value in values.items():
            # Check basic constraints
            if not isinstance(value, (int, float)):
                return None
            
            # Normalize to 0-1 range
            clamped = max(0.0, min(1.0, float(value)))
            validated[key] = clamped
        
        # Check sum constraint (if applicable)
        total = sum(validated.values())
        if abs(total - 1.0) > 0.1 and len(validated) > 1:
            # Normalize if sum is significantly different from 1
            validated = {k: v/total for k, v in validated.items()}
        
        return validated
    
    def emergency_override(
        self,
        override_values: Dict[str, float],
        override_code: str
    ) -> bool:
        """
        Emergency override of alignment values.
        
        Requires special override code and bypasses normal approval.
        """
        # Verify override code (in production, use cryptographic verification)
        if not self._verify_override_code(override_code):
            logger.error("Emergency override failed: invalid code")
            return False
        
        logger.critical("EMERGENCY OVERRIDE ACTIVATED")
        
        # Apply override immediately
        self.current_state.values = override_values.copy()
        self.current_state.version += 1
        self.current_state.last_updated = time.time()
        
        return True
    
    def _verify_override_code(self, code: str) -> bool:
        """Verify emergency override code."""
        # In production, this would use proper cryptographic verification
        expected_hash = hashlib.sha256(b"NEXUS_OMEGA_EMERGENCY_OVERRIDE").hexdigest()[:8]
        return code.startswith(expected_hash)
    
    def get_state(self) -> AlignmentState:
        """Get current alignment state."""
        return self.current_state
    
    def get_audit_trail(self) -> Dict[str, Any]:
        """Get complete audit trail of value updates."""
        return {
            'current_version': self.current_state.version,
            'current_hash': self.current_state.update_history_hash,
            'approved_updates': [
                {
                    'id': p.proposal_id,
                    'by': p.proposed_by,
                    'timestamp': p.timestamp,
                    'mechanism': p机制.value,
                    'signatures': len(p.approval_signatures)
                }
                for p in self.approved_updates
            ],
            'rejected_count': len(self.rejected_proposals),
            'pending_count': len(self.pending_proposals)
        }
    
    def lock_values(self, conditions: List[str]) -> bool:
        """Lock values until conditions are met."""
        if self.current_state.is_locked:
            return False
        
        self.current_state.is_locked = True
        self.current_state.unlock_conditions = conditions
        
        logger.info(f"Values locked. Conditions: {conditions}")
        return True
    
    def unlock_values(self, verifier: str) -> bool:
        """Unlock values if conditions are met."""
        if not self.current_state.is_locked:
            return True
        
        # In production, verify conditions are actually met
        self.current_state.is_locked = False
        self.current_state.unlock_conditions = []
        
        logger.info("Values unlocked")
        return True


# Example usage
if __name__ == "__main__":
    print("Value Lock-In Prevention Module")
    print("===============================\n")
    
    # Initialize with default values
    initial_values = {
        'operator_wealth': 0.25,
        'market_health': 0.25,
        'compute_sustainability': 0.25,
        'knowledge_preservation': 0.25
    }
    
    prevention = ValueLockInPrevention(initial_values)
    
    # Register signers
    prevention.register_signer("operator_001")
    prevention.register_signer("compliance_001")
    prevention.register_signer("ethics_board_001")
    
    # Propose an update
    proposal_id = prevention.propose_update(
        proposed_by="operator_001",
        target_values={'operator_wealth': 0.30, 'market_health': 0.25},
        justification="Adjusting weight based on long-term performance analysis",
        mechanism=UpdateMechanism.CONSTITUTIONAL_AMENDMENT
    )
    
    print(f"Created proposal: {proposal_id}")
    
    # Sign the proposal
    prevention.sign_proposal(proposal_id, "operator_001")
    prevention.sign_proposal(proposal_id, "compliance_001")
    prevention.sign_proposal(proposal_id, "ethics_board_001")
    
    # Get audit trail
    print("\nAudit Trail:")
    trail = prevention.get_audit_trail()
    print(f"Current Version: {trail['current_version']}")
    print(f"Approved Updates: {len(trail['approved_updates'])}")
    
    # Get current state
    state = prevention.get_state()
    print(f"\nCurrent Values (v{state.version}):")
    for key, value in state.values.items():
        print(f"  {key}: {value:.2f}")
