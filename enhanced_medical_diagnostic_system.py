from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Set, Tuple
import statistics
from dataclasses import dataclass, field
import json
from rapidfuzz import fuzz
from collections import defaultdict
import logging
from pathlib import Path

# ==================== ENHANCED ENUMS & CORE TYPES ====================
class BodySystem(Enum):
    CARDIOVASCULAR = 1
    NEUROLOGICAL = 2
    ENDOCRINE = 3 
    GASTROINTESTINAL = 4
    RESPIRATORY = 5
    MUSCULOSKELETAL = 6
    PSYCHIATRIC = 7
    DERMATOLOGICAL = 8
    HEMATOLOGICAL = 9
    IMMUNOLOGICAL = 10

class RiskLevel(Enum):
    CRITICAL = 4    # ER now
    HIGH = 3        # Urgent care <24h
    MODERATE = 2    # Primary care <1wk 
    LOW = 1         # Routine
    INFORMATIONAL = 0

class SymptomDuration(Enum):
    MINUTES = 1
    HOURS = 2
    DAYS_1_TO_3 = 3
    WEEKS_1_TO_2 = 4
    WEEKS_2_TO_4 = 5 
    MONTHS_1_TO_3 = 6
    MONTHS_3_PLUS = 7

class TemporalPattern(Enum):
    MORNING = 1
    EVENING = 2
    NIGHT = 3
    POST_PRANDIAL = 4
    EXERTIONAL = 5
    REST = 6
    INTERMITTENT = 7
    CONSTANT = 8

class ConditionRarity(Enum):
    COMMON = 1
    UNCOMMON = 2
    RARE = 3
    VERY_RARE = 4

# ==================== ENHANCED DATA STRUCTURES ====================
@dataclass
class UserProfile:
    age: int
    gender: str
    activity_level: str = "moderately_active"
    current_symptoms: List[str] = field(default_factory=list)
    medical_conditions: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    symptom_durations: Optional[Dict[str, SymptomDuration]] = None
    temporal_symptoms: Optional[Dict[str, TemporalPattern]] = None
    progression_data: Optional[Dict[str, str]] = None
    lab_values: Optional[Dict[str, float]] = None
    risk_factors: Set[str] = field(default_factory=set)
    family_history: Set[str] = field(default_factory=set)
    allergies: List[str] = field(default_factory=list)
    vital_signs: Optional[Dict[str, float]] = None

@dataclass
class MedicalCondition:
    icd11_code: str
    name: str
    body_systems: Set[BodySystem]
    prevalence: float  # 0-1
    symptom_weights: Dict[str, float]  # Symptom: 0-10 weight
    diagnostic_criteria: Set[str]
    risk_factors: Dict[str, float]  # {"smoking": 1.5}
    red_flag_combos: Set[frozenset]  # Critical symptom sets
    lab_thresholds: Dict[str, Tuple[str, float]]  # {"troponin": (">", 0.4)}
    rarity: ConditionRarity = ConditionRarity.COMMON
    severity: RiskLevel = RiskLevel.MODERATE

@dataclass 
class Diagnosis:
    condition: MedicalCondition
    confidence: float  # 0-100%
    matched_symptoms: Set[str]
    missing_criteria: Set[str]
    recommended_actions: List[str]
    urgency: RiskLevel
    evidence_summary: str
    differential_diagnoses: List[str] = field(default_factory=list)

@dataclass
class AdviceItem:
    category: str
    priority: RiskLevel
    title: str
    description: str
    recommendations: List[str]
    follow_up_required: bool = False
    timeline: str = "ongoing"
    evidence_level: str = "clinical"

@dataclass
class EmergencyLocation:
    name: str
    address: str
    distance_km: float
    phone: str
    type: str  # "ER", "Urgent Care", "Primary Care"

# ==================== KNOWLEDGE BASE ====================
class MedicalKnowledge:
    def __init__(self, knowledge_path: Optional[str] = None):
        if knowledge_path and Path(knowledge_path).exists():
            with open(knowledge_path) as f:
                data = json.load(f)
        else:
            data = self._get_default_knowledge()
        
        self.conditions = {}
        self.symptom_index = defaultdict(set)
        self.medication_effects = data.get("medication_effects", {})
        self.symptom_synonyms = data.get("symptom_synonyms", {})
        self.emergency_locations = data.get("emergency_locations", {})
        
        for cond_data in data["conditions"]:
            cond = MedicalCondition(
                icd11_code=cond_data["icd11"],
                name=cond_data["name"],
                body_systems={BodySystem[s] for s in cond_data["systems"]},
                prevalence=cond_data["prevalence"],
                symptom_weights=cond_data["symptoms"],
                diagnostic_criteria=set(cond_data["criteria"]),
                risk_factors=cond_data["risk_factors"],
                red_flag_combos={frozenset(s) for s in cond_data["red_flags"]},
                lab_thresholds=cond_data.get("labs", {}),
                rarity=ConditionRarity[cond_data.get("rarity", "COMMON")],
                severity=RiskLevel[cond_data.get("severity", "MODERATE")]
            )
            self.conditions[cond.icd11_code] = cond
            
            for symptom in cond.symptom_weights:
                self.symptom_index[symptom].add(cond.icd11_code)
    
    def _get_default_knowledge(self) -> Dict:
        """Default knowledge base with comprehensive medical data"""
        return {
            "conditions": [
                {
                    "icd11": "I21.9",
                    "name": "Acute myocardial infarction",
                    "systems": ["CARDIOVASCULAR"],
                    "prevalence": 0.02,
                    "symptoms": {
                        "chest_pain": 9.5,
                        "radiating_arm_pain": 8.2,
                        "diaphoresis": 7.1,
                        "shortness_of_breath": 8.0,
                        "nausea": 6.5,
                        "fatigue": 6.0
                    },
                    "criteria": ["chest_pain", "diaphoresis"],
                    "risk_factors": {
                        "age>45": 2.1,
                        "diabetes": 1.8,
                        "smoking": 1.5,
                        "hypertension": 1.4
                    },
                    "red_flags": [
                        ["chest_pain", "diaphoresis", "shortness_of_breath"]
                    ],
                    "labs": {
                        "troponin": [">", 0.4],
                        "bnp": [">", 500]
                    },
                    "rarity": "COMMON",
                    "severity": "CRITICAL"
                },
                {
                    "icd11": "E11.9",
                    "name": "Type 2 diabetes mellitus",
                    "systems": ["ENDOCRINE"],
                    "prevalence": 0.08,
                    "symptoms": {
                        "excessive_thirst": 7.5,
                        "frequent_urination": 7.0,
                        "unexplained_weight_loss": 6.5,
                        "fatigue": 6.0,
                        "blurred_vision": 5.5,
                        "slow_healing": 5.0
                    },
                    "criteria": ["excessive_thirst", "frequent_urination"],
                    "risk_factors": {
                        "age>40": 1.8,
                        "obesity": 2.2,
                        "family_history": 1.6
                    },
                    "red_flags": [],
                    "labs": {
                        "glucose": [">", 200],
                        "hba1c": [">", 6.5]
                    },
                    "rarity": "COMMON",
                    "severity": "HIGH"
                },
                {
                    "icd11": "I10",
                    "name": "Essential hypertension",
                    "systems": ["CARDIOVASCULAR"],
                    "prevalence": 0.25,
                    "symptoms": {
                        "headaches": 6.0,
                        "dizziness": 5.5,
                        "chest_pain": 7.0,
                        "shortness_of_breath": 6.5,
                        "vision_problems": 5.0
                    },
                    "criteria": ["headaches", "dizziness"],
                    "risk_factors": {
                        "age>50": 1.5,
                        "obesity": 1.8,
                        "family_history": 1.4
                    },
                    "red_flags": [],
                    "labs": {
                        "systolic_bp": [">", 140],
                        "diastolic_bp": [">", 90]
                    },
                    "rarity": "COMMON",
                    "severity": "HIGH"
                },
                {
                    "icd11": "J45.9",
                    "name": "Asthma",
                    "systems": ["RESPIRATORY"],
                    "prevalence": 0.07,
                    "symptoms": {
                        "wheezing": 8.0,
                        "shortness_of_breath": 7.5,
                        "cough": 6.0,
                        "chest_tightness": 7.0
                    },
                    "criteria": ["wheezing", "shortness_of_breath"],
                    "risk_factors": {
                        "allergies": 1.8,
                        "family_history": 1.6,
                        "smoking": 1.3
                    },
                    "red_flags": [],
                    "labs": {},
                    "rarity": "COMMON",
                    "severity": "MODERATE"
                }
            ],
            "medication_effects": {
                "beta_blockers": {
                    "masks": ["tachycardia", "palpitations"],
                    "causes": ["fatigue", "depression", "cold_extremities"]
                },
                "steroids": {
                    "causes": ["mood_changes", "weight_gain", "insomnia"]
                },
                "diuretics": {
                    "causes": ["frequent_urination", "dehydration"]
                }
            },
            "symptom_synonyms": {
                "chest_pain": ["heart pain", "tight chest", "pressure in chest"],
                "diaphoresis": ["cold sweat", "excessive sweating"],
                "shortness_of_breath": ["difficulty breathing", "breathlessness"],
                "excessive_thirst": ["polydipsia", "always thirsty"],
                "frequent_urination": ["polyuria", "urinating often"],
                "wheezing": ["whistling sound", "breathing difficulty"]
            },
            "emergency_locations": {
                "default_er": {
                    "name": "General Hospital Emergency Room",
                    "address": "123 Medical Center Dr",
                    "phone": "911",
                    "type": "ER"
                }
            }
        }
    
    def get_condition(self, code: str) -> Optional[MedicalCondition]:
        return self.conditions.get(code)
    
    def get_conditions_for_symptoms(self, symptoms: Set[str]) -> Set[str]:
        codes = set()
        for s in symptoms:
            codes.update(self.symptom_index.get(s, set()))
        return codes

# ==================== SYMPTOM NORMALIZER ====================
class SymptomNormalizer:
    """Converts layman's terms to medical symptom keys with fuzzy matching"""
    def __init__(self, synonym_map: Dict[str, List[str]]):
        self.synonym_map = synonym_map
        self.reverse_map = {}
        for std_term, variants in synonym_map.items():
            for var in variants:
                self.reverse_map[var.lower()] = std_term
    
    def normalize(self, term: str) -> Optional[str]:
        term_lower = term.lower().strip()
        
        # Exact match check
        if term_lower in self.reverse_map:
            return self.reverse_map[term_lower]
        
        # Fuzzy match
        best_match, best_score = None, 0
        for variant, std_term in self.reverse_map.items():
            score = fuzz.ratio(term_lower, variant)
            if score > best_score and score > 75:  # 75% similarity threshold
                best_match, best_score = std_term, score
        
        return best_match

# ==================== ENHANCED CORE ANALYZER ====================
class EnhancedSymptomAnalyzer:
    """Advanced medical diagnostic system with comprehensive analysis"""
    
    def __init__(self, knowledge_base: MedicalKnowledge):
        self.knowledge = knowledge_base
        self.normalizer = SymptomNormalizer(knowledge_base.symptom_synonyms)
        self.logger = logging.getLogger(__name__)
        
        # Feedback learning system
        self.diagnosis_feedback = defaultdict(list)
        
    def evaluate_user(self, user: UserProfile) -> Tuple[List[Diagnosis], List[str]]:
        """Main analysis pipeline with enhanced features"""
        try:
            # Step 1: Preprocess and normalize inputs
            normalized_symptoms = self._normalize_symptoms(user.current_symptoms)
            
            # Step 2: Check critical red flags
            red_flags = self._check_red_flags(normalized_symptoms)
            
            # Step 3: Identify candidate conditions
            candidate_codes = self.knowledge.get_conditions_for_symptoms(set(normalized_symptoms))
            
            # Step 4: Score each condition
            diagnoses = []
            for code in candidate_codes:
                cond = self.knowledge.get_condition(code)
                if not cond:
                    continue
                    
                diagnosis = self._assess_condition(cond, normalized_symptoms, user)
                if diagnosis.confidence > 15:  # Lower threshold for more comprehensive results
                    diagnoses.append(diagnosis)
            
            # Step 5: Prioritize and filter results
            diagnoses.sort(key=lambda d: (-d.confidence, -d.urgency.value))
            
            # Step 6: Generate differential diagnoses
            self._add_differential_diagnoses(diagnoses)
            
            return diagnoses[:10], red_flags
            
        except Exception as e:
            self.logger.error(f"Error in evaluation: {e}")
            return [], []
    
    def _normalize_symptoms(self, symptoms: List[str]) -> Dict[str, Tuple[float, Optional[SymptomDuration]]]:
        """Normalize symptoms and assign default severity"""
        normalized = {}
        for symptom in symptoms:
            normalized_symptom = self.normalizer.normalize(symptom)
            if normalized_symptom:
                normalized[normalized_symptom] = (7.0, None)  # Default severity
        return normalized
    
    def _assess_condition(self, cond: MedicalCondition, 
                         symptoms: Dict[str, Tuple[float, Optional[SymptomDuration]]],
                         user: UserProfile) -> Diagnosis:
        """Calculate comprehensive confidence score for a condition"""
        
        # Base symptom matching
        matched = set()
        symptom_score = 0.0
        total_weight = 0.0
        
        for symptom, (severity, duration) in symptoms.items():
            if symptom in cond.symptom_weights:
                matched.add(symptom)
                weight = cond.symptom_weights[symptom]
                
                # Adjust for severity and duration
                adjusted_weight = weight * severity
                
                if duration:
                    duration_modifiers = {
                        SymptomDuration.MINUTES: 1.5,
                        SymptomDuration.HOURS: 1.3,
                        SymptomDuration.DAYS_1_TO_3: 1.1,
                        SymptomDuration.WEEKS_1_TO_2: 1.0,
                        SymptomDuration.WEEKS_2_TO_4: 0.9,
                        SymptomDuration.MONTHS_1_TO_3: 0.8,
                        SymptomDuration.MONTHS_3_PLUS: 0.7
                    }
                    adjusted_weight *= duration_modifiers.get(duration, 1.0)
                
                symptom_score += adjusted_weight
                total_weight += weight
        
        # Normalize symptom score
        if total_weight > 0:
            symptom_score /= total_weight
        
        # Risk factor adjustment
        risk_modifier = 1.0
        for factor, multiplier in cond.risk_factors.items():
            if self._check_risk_factor(factor, user):
                risk_modifier *= multiplier
        
        # Lab test confirmation
        lab_modifier = 1.0
        if user.lab_values:
            lab_modifier = self._calculate_lab_modifier(cond, user.lab_values)
        
        # Medication interaction adjustment
        med_modifier = self._calculate_medication_modifier(user, cond)
        
        # Temporal pattern adjustment
        temporal_modifier = 1.0
        if user.temporal_symptoms:
            temporal_modifier = self._calculate_temporal_modifier(cond, user.temporal_symptoms)
        
        # Final confidence calculation
        confidence = min(100, (
            symptom_score * 100 * 
            cond.prevalence * 10 * 
            risk_modifier * 
            lab_modifier * 
            med_modifier * 
            temporal_modifier
        ))
        
        # Determine urgency
        urgency = self._determine_urgency(cond, confidence, matched, user)
        
        # Generate recommendations
        actions = self._generate_actions(cond, confidence, urgency, user)
        
        # Create evidence summary
        evidence_summary = self._create_evidence_summary(cond, matched, confidence)
        
        return Diagnosis(
            condition=cond,
            confidence=confidence,
            matched_symptoms=matched,
            missing_criteria=cond.diagnostic_criteria - matched,
            recommended_actions=actions,
            urgency=urgency,
            evidence_summary=evidence_summary
        )
    
    def _check_risk_factor(self, factor: str, user: UserProfile) -> bool:
        """Check if user has a specific risk factor"""
        if factor.startswith("age>"):
            age_threshold = int(factor.split(">")[1])
            return user.age > age_threshold
        elif factor in user.risk_factors:
            return True
        elif factor in user.family_history:
            return True
        elif factor in user.medical_conditions:
            return True
        return False
    
    def _calculate_lab_modifier(self, cond: MedicalCondition, lab_values: Dict[str, float]) -> float:
        """Calculate modifier based on lab results"""
        modifier = 1.0
        for test, (op, val) in cond.lab_thresholds.items():
            if test in lab_values:
                actual = lab_values[test]
                if op == ">" and actual > val:
                    modifier *= 1.3
                elif op == "<" and actual < val:
                    modifier *= 1.3
                elif op == "==" and actual == val:
                    modifier *= 1.2
        return modifier
    
    def _calculate_medication_modifier(self, user: UserProfile, cond: MedicalCondition) -> float:
        """Calculate modifier based on medication interactions"""
        modifier = 1.0
        for med in user.current_medications:
            if med in self.knowledge.medication_effects:
                effects = self.knowledge.medication_effects[med]
                # Check if medication masks symptoms of this condition
                if "masks" in effects:
                    for masked_symptom in effects["masks"]:
                        if masked_symptom in cond.symptom_weights:
                            modifier *= 0.8  # Reduce confidence if symptoms are masked
        return modifier
    
    def _calculate_temporal_modifier(self, cond: MedicalCondition, temporal_data: Dict[str, TemporalPattern]) -> float:
        """Calculate modifier based on temporal patterns"""
        # This could be expanded with condition-specific temporal patterns
        return 1.0
    
    def _determine_urgency(self, cond: MedicalCondition, confidence: float, 
                          matched_symptoms: Set[str], user: UserProfile) -> RiskLevel:
        """Determine urgency level based on condition and symptoms"""
        # Check for red flag combinations
        for combo in cond.red_flag_combos:
            if combo.issubset(matched_symptoms):
                return RiskLevel.CRITICAL
        
        # Base urgency on condition severity
        if cond.severity == RiskLevel.CRITICAL:
            return RiskLevel.CRITICAL
        elif cond.severity == RiskLevel.HIGH and confidence > 60:
            return RiskLevel.HIGH
        elif confidence > 80:
            return RiskLevel.HIGH
        elif confidence > 50:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _generate_actions(self, cond: MedicalCondition, confidence: float, 
                         urgency: RiskLevel, user: UserProfile) -> List[str]:
        """Generate recommended actions based on diagnosis"""
        actions = []
        
        if urgency == RiskLevel.CRITICAL:
            actions.extend([
                "Seek immediate emergency medical care",
                "Call emergency services if symptoms worsen",
                "Do not delay treatment"
            ])
        elif urgency == RiskLevel.HIGH:
            actions.extend([
                f"Schedule urgent evaluation for {cond.name}",
                "Contact healthcare provider within 24 hours",
                "Monitor symptoms closely"
            ])
        elif urgency == RiskLevel.MODERATE:
            actions.extend([
                f"Consider evaluation for {cond.name}",
                "Schedule appointment within 1 week",
                "Track symptom progression"
            ])
        else:
            actions.extend([
                f"Routine evaluation for {cond.name}",
                "Schedule appointment within 2-4 weeks",
                "Continue monitoring"
            ])
        
        # Add condition-specific recommendations
        if cond.icd11_code == "I21.9":  # Heart attack
            actions.append("Avoid physical exertion")
        elif cond.icd11_code == "E11.9":  # Diabetes
            actions.append("Monitor blood glucose levels")
        
        return actions
    
    def _create_evidence_summary(self, cond: MedicalCondition, matched_symptoms: Set[str], 
                                confidence: float) -> str:
        """Create human-readable evidence summary"""
        symptom_count = len(matched_symptoms)
        total_symptoms = len(cond.symptom_weights)
        match_percentage = (symptom_count / total_symptoms * 100) if total_symptoms > 0 else 0
        
        return (f"{symptom_count}/{total_symptoms} key symptoms match this condition "
                f"({match_percentage:.1f}% match rate). Confidence: {confidence:.1f}%")
    
    def _add_differential_diagnoses(self, diagnoses: List[Diagnosis]):
        """Add differential diagnoses to each diagnosis"""
        for dx in diagnoses:
            differentials = []
            for other_dx in diagnoses:
                if other_dx.condition.icd11_code != dx.condition.icd11_code:
                    # Check if conditions share body systems
                    if dx.condition.body_systems & other_dx.condition.body_systems:
                        differentials.append(other_dx.condition.name)
            dx.differential_diagnoses = differentials[:3]  # Top 3 differentials
    
    def _check_red_flags(self, symptoms: Dict[str, Tuple[float, Optional[SymptomDuration]]]) -> List[str]:
        """Check for critical red flag symptoms"""
        red_flags = []
        
        # Check individual red flag symptoms
        critical_symptoms = {
            "chest_pain": "Possible cardiac event - seek immediate care",
            "shortness_of_breath": "Respiratory distress - urgent evaluation needed",
            "unexplained_weight_loss": "Possible malignancy or serious metabolic disorder",
            "numbness_tingling": "Possible neurological emergency if sudden onset",
            "severe_headache": "Possible stroke or aneurysm - seek immediate care"
        }
        
        for symptom in symptoms:
            if symptom in critical_symptoms:
                red_flags.append(f"{symptom}: {critical_symptoms[symptom]}")
        
        # Check red flag combinations
        for cond in self.knowledge.conditions.values():
            for combo in cond.red_flag_combos:
                if combo.issubset(set(symptoms.keys())):
                    red_flags.append(f"Critical combination: {', '.join(combo)} - {cond.name}")
        
        return red_flags
    
    def log_diagnostic_feedback(self, diagnosis: Diagnosis, was_correct: bool, 
                               actual_condition: Optional[str] = None):
        """Log feedback for learning and improvement"""
        feedback = {
            "diagnosis": diagnosis.condition.icd11_code,
            "confidence": diagnosis.confidence,
            "was_correct": was_correct,
            "actual_condition": actual_condition,
            "timestamp": datetime.now().isoformat()
        }
        self.diagnosis_feedback[diagnosis.condition.icd11_code].append(feedback)
    
    def get_nearest_emergency_location(self, diagnosis: Diagnosis, zip_code: str = None) -> EmergencyLocation:
        """Get nearest emergency location based on diagnosis urgency"""
        if diagnosis.urgency == RiskLevel.CRITICAL:
            # In a real implementation, this would use geolocation APIs
            return EmergencyLocation(
                name="Nearest Emergency Room",
                address="Use GPS or call 911",
                distance_km=0.0,
                phone="911",
                type="ER"
            )
        else:
            return EmergencyLocation(
                name="Nearest Urgent Care",
                address="Use GPS or call local urgent care",
                distance_km=0.0,
                phone="Local urgent care",
                type="Urgent Care"
            )

# ==================== ENHANCED PUBLIC INTERFACE ====================
def analyze_symptoms_and_suggest_conditions(
    symptoms: List[str], 
    age: int,
    gender: str,
    existing_conditions: List[str] = None,
    medications: List[str] = None,
    symptom_durations: Optional[Dict[str, str]] = None,
    temporal_data: Optional[Dict[str, str]] = None,
    lab_values: Optional[Dict[str, float]] = None,
    risk_factors: Set[str] = None,
    family_history: Set[str] = None
) -> Dict:
    """
    Enhanced public function to analyze symptoms and suggest possible conditions
    
    Args:
        symptoms: List of symptom descriptions (can be layman's terms)
        age: Patient age
        gender: Patient gender
        existing_conditions: List of already diagnosed conditions
        medications: List of current medications
        symptom_durations: Dict mapping symptom keys to duration strings 
        temporal_data: Dict mapping temporal patterns to their occurrences
        lab_values: Dict mapping lab test names to their values
        risk_factors: Set of risk factors
        family_history: Set of family medical history
    """
    # Initialize knowledge base and analyzer
    knowledge = MedicalKnowledge()
    analyzer = EnhancedSymptomAnalyzer(knowledge)
    
    # Convert duration strings to enum if provided
    duration_enums = None
    if symptom_durations:
        duration_enums = {}
        for symptom, duration in symptom_durations.items():
            try:
                duration_enums[symptom] = SymptomDuration[duration]
            except KeyError:
                continue
    
    # Convert temporal strings to enum if provided
    temporal_enums = None
    if temporal_data:
        temporal_enums = {}
        for pattern, temporal in temporal_data.items():
            try:
                temporal_enums[pattern] = TemporalPattern[temporal]
            except KeyError:
                continue
    
    # Create user profile
    user = UserProfile(
        age=age,
        gender=gender,
        current_symptoms=symptoms,
        medical_conditions=existing_conditions or [],
        current_medications=medications or [],
        symptom_durations=duration_enums,
        temporal_symptoms=temporal_enums,
        lab_values=lab_values,
        risk_factors=risk_factors or set(),
        family_history=family_history or set()
    )
    
    # Perform analysis
    diagnoses, red_flags = analyzer.evaluate_user(user)
    
    # Convert diagnoses to serializable format
    diagnosis_results = []
    for dx in diagnoses:
        diagnosis_results.append({
            "condition_name": dx.condition.name,
            "icd11_code": dx.condition.icd11_code,
            "confidence": dx.confidence,
            "urgency": dx.urgency.name,
            "matched_symptoms": list(dx.matched_symptoms),
            "missing_criteria": list(dx.missing_criteria),
            "recommended_actions": dx.recommended_actions,
            "evidence_summary": dx.evidence_summary,
            "differential_diagnoses": dx.differential_diagnoses,
            "body_systems": [system.name for system in dx.condition.body_systems]
        })
    
    return {
        "symptoms": symptoms,
        "symptom_durations": symptom_durations,
        "temporal_data": temporal_data,
        "lab_values": lab_values,
        "possible_conditions": diagnosis_results,
        "red_flags": red_flags,
        "has_red_flags": len(red_flags) > 0,
        "timestamp": datetime.now().isoformat(),
        "analysis_metadata": {
            "total_conditions_evaluated": len(diagnoses),
            "knowledge_base_version": "1.0",
            "analyzer_version": "2.0"
        }
    }

# ==================== FASTAPI INTEGRATION ====================
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List as PydanticList
    
    class SymptomAnalysisRequest(BaseModel):
        symptoms: PydanticList[str]
        age: int
        gender: str
        existing_conditions: PydanticList[str] = []
        medications: PydanticList[str] = []
        symptom_durations: Optional[Dict[str, str]] = None
        temporal_data: Optional[Dict[str, str]] = None
        lab_values: Optional[Dict[str, float]] = None
        risk_factors: PydanticList[str] = []
        family_history: PydanticList[str] = []
    
    class SymptomAnalysisResponse(BaseModel):
        possible_conditions: List[Dict]
        red_flags: List[str]
        has_red_flags: bool
        timestamp: str
        analysis_metadata: Dict
    
    app = FastAPI(title="Enhanced Medical Diagnostic System", version="2.0")
    
    # Initialize analyzer globally
    knowledge_base = MedicalKnowledge()
    diagnostic_system = EnhancedSymptomAnalyzer(knowledge_base)
    
    @app.post("/diagnose", response_model=SymptomAnalysisResponse)
    async def diagnose_symptoms(request: SymptomAnalysisRequest):
        """Analyze symptoms and provide diagnostic suggestions"""
        try:
            result = analyze_symptoms_and_suggest_conditions(
                symptoms=request.symptoms,
                age=request.age,
                gender=request.gender,
                existing_conditions=request.existing_conditions,
                medications=request.medications,
                symptom_durations=request.symptom_durations,
                temporal_data=request.temporal_data,
                lab_values=request.lab_values,
                risk_factors=set(request.risk_factors),
                family_history=set(request.family_history)
            )
            return SymptomAnalysisResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "version": "2.0"}
    
    @app.get("/conditions")
    async def list_conditions():
        """List all available conditions in the knowledge base"""
        conditions = []
        for code, cond in knowledge_base.conditions.items():
            conditions.append({
                "icd11_code": code,
                "name": cond.name,
                "body_systems": [system.name for system in cond.body_systems],
                "prevalence": cond.prevalence,
                "severity": cond.severity.name
            })
        return {"conditions": conditions}
    
    @app.get("/symptoms")
    async def list_symptoms():
        """List all available symptoms in the knowledge base"""
        symptoms = list(knowledge_base.symptom_index.keys())
        return {"symptoms": symptoms}
    
except ImportError:
    # FastAPI not available, skip API endpoints
    pass

# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Example usage with enhanced features
    print("Enhanced Medical Diagnostic System v2.0")
    print("=" * 50)
    
    # Test case 1: Cardiac symptoms
    test_symptoms = ["chest pain", "shortness of breath", "cold sweat"]
    test_durations = {
        "chest_pain": "MINUTES",
        "shortness_of_breath": "HOURS"
    }
    
    test_labs = {
        "troponin": 0.6,
        "bnp": 600
    }
    
    print(f"Analyzing symptoms: {', '.join(test_symptoms)}")
    print(f"With durations: {test_durations}")
    print(f"With lab values: {test_labs}")
    
    analysis = analyze_symptoms_and_suggest_conditions(
        symptoms=test_symptoms,
        age=55,
        gender="M",
        existing_conditions=["hypertension"],
        medications=["lisinopril"],
        symptom_durations=test_durations,
        lab_values=test_labs,
        risk_factors={"smoking", "family_history_cvd"}
    )
    
    # Display results
    if analysis["has_red_flags"]:
        print("\n🚨 RED FLAG WARNINGS:")
        for flag in analysis["red_flags"]:
            print(f"- {flag}")
    
    print("\nTOP DIAGNOSES:")
    for i, condition in enumerate(analysis["possible_conditions"][:3], 1):
        print(f"\n{i}. {condition['condition_name']} (ICD11: {condition['icd11_code']})")
        print(f"   🔍 Confidence: {condition['confidence']:.1f}%")
        print(f"   ⚠️ Urgency: {condition['urgency']}")
        print(f"   📋 Evidence: {condition['evidence_summary']}")
        print(f"   💊 Recommended actions:")
        for action in condition['recommended_actions'][:3]:
            print(f"     • {action}")
        if condition['differential_diagnoses']:
            print(f"   🔄 Differential diagnoses: {', '.join(condition['differential_diagnoses'])}")
    
    print("\n" + "=" * 50)
    print("Note: This is not a diagnosis. Please consult a healthcare professional for evaluation.")
    print("For emergency symptoms, call 911 immediately.") 