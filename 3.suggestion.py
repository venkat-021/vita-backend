import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class ActivityLevel(Enum):
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTREMELY_ACTIVE = "extremely_active"

class SpecialistType(Enum):
    """Medical specialists for referrals"""
    CARDIOLOGIST = "cardiologist"
    ENDOCRINOLOGIST = "endocrinologist"
    ORTHOPEDIST = "orthopedist"
    NEPHROLOGIST = "nephrologist"
    GASTROENTEROLOGIST = "gastroenterologist"
    PSYCHIATRIST = "psychiatrist"
    PULMONOLOGIST = "pulmonologist"
    RHEUMATOLOGIST = "rheumatologist"
    NEUROLOGIST = "neurologist"
    ONCOLOGIST = "oncologist"
    DERMATOLOGIST = "dermatologist"
    OPHTHALMOLOGIST = "ophthalmologist"
    OTOLARYNGOLOGIST = "otolaryngologist"  # ENT
    GYNECOLOGIST = "gynecologist"
    UROLOGIST = "urologist"
    DIETITIAN = "registered_dietitian"
    PHYSICAL_THERAPIST = "physical_therapist"
    MENTAL_HEALTH_COUNSELOR = "mental_health_counselor"
    SLEEP_SPECIALIST = "sleep_specialist"
    PAIN_MANAGEMENT = "pain_management_specialist"

@dataclass
class DoctorRecommendation:
    specialist_type: SpecialistType
    urgency: RiskLevel
    reason: str
    symptoms_indicators: List[str]
    recommended_tests: List[str] = field(default_factory=list)
    timeline: str = "within 2-4 weeks"
    notes: str = ""

@dataclass
class MedicalCondition:
    name: str
    severity: RiskLevel
    restrictions: List[str] = field(default_factory=list)
    modifications: List[str] = field(default_factory=list)
    monitoring_required: bool = False
    specialist_required: Optional[SpecialistType] = None

@dataclass
class UserProfile:
    age: int
    gender: str
    activity_level: ActivityLevel
    medical_conditions: List[MedicalCondition] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    pregnancy_status: bool = False
    smoking_status: bool = False
    alcohol_consumption: str = "none"  # none, light, moderate, heavy
    sleep_quality: int = 7  # 1-10 scale
    stress_level: int = 5  # 1-10 scale
    family_history: Dict[str, bool] = field(default_factory=dict)
    recent_bloodwork: Dict[str, Any] = field(default_factory=dict)
    fitness_goals: List[str] = field(default_factory=list)
    dietary_preferences: List[str] = field(default_factory=list)
    # New fields for symptoms
    current_symptoms: List[str] = field(default_factory=list)
    pain_locations: List[str] = field(default_factory=list)
    chronic_fatigue: bool = False
    frequent_headaches: bool = False
    digestive_issues: bool = False
    breathing_difficulties: bool = False
    skin_problems: bool = False
    vision_problems: bool = False
    hearing_problems: bool = False
    joint_pain: bool = False
    muscle_weakness: bool = False
    mood_changes: bool = False

@dataclass
class BodyMetrics:
    bmi: float
    whr: float
    whtr: float
    body_fat_percentage: float
    muscle_mass_percentage: float
    waist_cm: float
    arm_to_leg_ratio: float
    resting_heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None

@dataclass
class AdviceItem:
    category: str
    priority: RiskLevel
    title: str
    description: str
    recommendations: List[str]
    restrictions: List[str] = field(default_factory=list)
    medical_notes: List[str] = field(default_factory=list)
    follow_up_required: bool = False
    timeline: str = "ongoing"
    doctor_recommendations: List[DoctorRecommendation] = field(default_factory=list)

class MedicalConditionManager:
    """Manages medical conditions and their impact on health advice"""

    CONDITION_DATABASE = {
        'diabetes_type1': MedicalCondition(
            name='Type 1 Diabetes',
            severity=RiskLevel.HIGH,
            restrictions=['no_extreme_calorie_restriction', 'monitor_blood_sugar', 'no_fasting'],
            modifications=['frequent_meals', 'complex_carbs_preferred', 'insulin_timing_coordination'],
            monitoring_required=True
        ),
        'diabetes_type2': MedicalCondition(
            name='Type 2 Diabetes',
            severity=RiskLevel.HIGH,
            restrictions=['limit_simple_sugars', 'monitor_blood_sugar', 'gradual_exercise_increase'],
            modifications=['low_glycemic_diet', 'portion_control', 'regular_meal_timing'],
            monitoring_required=True
        ),
        'hypertension': MedicalCondition(
            name='Hypertension',
            severity=RiskLevel.HIGH,
            restrictions=['limit_sodium', 'avoid_intense_isometric_exercises', 'monitor_blood_pressure'],
            modifications=['dash_diet', 'gradual_exercise_progression', 'stress_management'],
            monitoring_required=True
        ),
        'heart_disease': MedicalCondition(
            name='Cardiovascular Disease',
            severity=RiskLevel.CRITICAL,
            restrictions=['no_high_intensity_without_clearance', 'limit_sodium', 'monitor_heart_rate'],
            modifications=['cardiac_rehabilitation_exercises', 'mediterranean_diet', 'stress_reduction'],
            monitoring_required=True
        ),
        'osteoporosis': MedicalCondition(
            name='Osteoporosis',
            severity=RiskLevel.MODERATE,
            restrictions=['avoid_high_impact_exercises', 'avoid_forward_spinal_flexion'],
            modifications=['weight_bearing_exercises', 'balance_training', 'calcium_vitamin_d_focus'],
            monitoring_required=False
        ),
        'arthritis': MedicalCondition(
            name='Arthritis',
            severity=RiskLevel.MODERATE,
            restrictions=['avoid_high_impact_exercises', 'joint_protection'],
            modifications=['low_impact_exercises', 'anti_inflammatory_diet', 'flexibility_focus'],
            monitoring_required=False
        ),
        'kidney_disease': MedicalCondition(
            name='Chronic Kidney Disease',
            severity=RiskLevel.HIGH,
            restrictions=['limit_protein', 'limit_potassium', 'limit_phosphorus', 'monitor_fluid_intake'],
            modifications=['renal_diet', 'gentle_exercise', 'blood_pressure_control'],
            monitoring_required=True
        ),
        'eating_disorder': MedicalCondition(
            name='Eating Disorder History',
            severity=RiskLevel.HIGH,
            restrictions=['no_calorie_counting', 'no_restrictive_diets', 'no_weight_focus'],
            modifications=['intuitive_eating', 'mental_health_support', 'balanced_approach'],
            monitoring_required=True
        ),
        'pregnancy': MedicalCondition(
            name='Pregnancy',
            severity=RiskLevel.HIGH,
            restrictions=['avoid_alcohol', 'limit_caffeine', 'avoid_high_risk_activities'],
            modifications=['prenatal_nutrition', 'safe_exercise', 'regular_prenatal_care'],
            monitoring_required=True
        ),
        'asthma': MedicalCondition(
            name='Asthma',
            severity=RiskLevel.MODERATE,
            restrictions=['avoid_trigger_environments', 'carry_inhaler'],
            modifications=['breathing_exercises', 'gradual_cardio_build_up'],
            monitoring_required=True
        ),
        'thyroid_disorder': MedicalCondition(
            name='Thyroid Disorder',
            severity=RiskLevel.MODERATE,
            restrictions=['consistent_medication_timing'],
            modifications=['metabolism-focused_nutrition', 'energy_management'],
            monitoring_required=True
        )
    }

    @classmethod
    def get_condition(cls, condition_key: str) -> Optional[MedicalCondition]:
        return cls.CONDITION_DATABASE.get(condition_key.lower())

    @classmethod
    def assess_overall_risk(cls, conditions: List[MedicalCondition]) -> RiskLevel:
        if not conditions:
            return RiskLevel.LOW
        max_severity = max(condition.severity for condition in conditions)
        return max_severity

class AdvancedBodyAnalyzer:
    """Advanced body composition analysis with medical considerations"""

    @staticmethod
    def calculate_improved_body_fat(bmi: float, age: int, gender: str,
                                  activity_level: ActivityLevel) -> float:
        """Improved body fat calculation using multiple factors"""
        base_bfp = (1.20 * bmi) + (0.23 * age) - (16.2 if gender.upper() == 'M' else 5.4)

        # Activity level adjustment
        activity_adjustments = {
            ActivityLevel.SEDENTARY: 2,
            ActivityLevel.LIGHTLY_ACTIVE: 1,
            ActivityLevel.MODERATELY_ACTIVE: 0,
            ActivityLevel.VERY_ACTIVE: -1,
            ActivityLevel.EXTREMELY_ACTIVE: -2
        }

        base_bfp += activity_adjustments.get(activity_level, 0)
        return max(5, min(50, base_bfp))  # Reasonable bounds

    @staticmethod
    def calculate_muscle_mass(bmi: float, age: int, gender: str, activity_level: ActivityLevel) -> float:
        """More accurate muscle mass calculation"""
        base_mm = (0.85 * (100 - AdvancedBodyAnalyzer.calculate_improved_body_fat(bmi, age, gender, activity_level)))

        # Gender adjustment
        if gender.upper() == 'M':
            base_mm += 5  # Men typically have higher muscle mass

        # Age adjustment
        if age > 50:
            base_mm -= (age - 50) * 0.2  # Account for sarcopenia

        return max(30, min(60, base_mm))  # Reasonable bounds

    @staticmethod
    def assess_metabolic_risk(metrics: BodyMetrics, user: UserProfile) -> RiskLevel:
        """Assess metabolic syndrome risk"""
        risk_factors = 0

        # Waist circumference
        waist_threshold = 102 if user.gender.upper() == 'M' else 88
        if metrics.waist_cm >= waist_threshold:
            risk_factors += 1

        # WHR
        whr_threshold = 0.90 if user.gender.upper() == 'M' else 0.85
        if metrics.whr >= whr_threshold:
            risk_factors += 1

        # WHtR
        if metrics.whtr >= 0.5:
            risk_factors += 1

        # BMI
        if metrics.bmi >= 30:
            risk_factors += 1

        # Age factor
        if user.age >= 45:
            risk_factors += 1

        if risk_factors >= 3:
            return RiskLevel.HIGH
        elif risk_factors == 2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

class DoctorRecommendationEngine:
    """Engine to recommend appropriate medical specialists"""
    
    SPECIALIST_DATABASE = {
        # Cardiovascular specialists
        SpecialistType.CARDIOLOGIST: {
            "name": "Cardiologist (Heart Specialist)",
            "conditions": ["heart_disease", "hypertension", "chest_pain", "arrhythmia", "heart_failure"],
            "symptoms": ["chest_pain", "shortness_of_breath", "irregular_heartbeat", "leg_swelling"],
            "risk_factors": ["high_cholesterol", "diabetes", "family_heart_disease", "smoking"],
            "tests": ["ECG", "Echocardiogram", "Stress Test", "Cardiac Catheterization", "Lipid Profile"]
        },
        # Endocrine specialists
        SpecialistType.ENDOCRINOLOGIST: {
            "name": "Endocrinologist (Hormone Specialist)",
            "conditions": ["diabetes_type1", "diabetes_type2", "thyroid_disorder", "metabolic_syndrome"],
            "symptoms": ["excessive_thirst", "frequent_urination", "fatigue", "weight_changes"],
            "risk_factors": ["family_diabetes", "obesity", "sedentary_lifestyle"],
            "tests": ["HbA1c", "Glucose Tolerance Test", "Thyroid Function Tests", "Insulin Levels"]
        },
        # Orthopedic specialists
        SpecialistType.ORTHOPEDIST: {
            "name": "Orthopedist (Bone and Joint Specialist)",
            "conditions": ["arthritis", "osteoporosis", "joint_pain", "back_pain"],
            "symptoms": ["joint_pain", "stiffness", "reduced_mobility", "bone_pain"],
            "risk_factors": ["age_over_50", "previous_fractures", "family_osteoporosis"],
            "tests": ["Bone Density Scan", "X-rays", "MRI", "Joint Aspiration"]
        }
    }
    
    @classmethod
    def analyze_symptoms_and_recommend(cls, user: UserProfile, metrics: BodyMetrics) -> List[DoctorRecommendation]:
        """Analyze user profile and recommend appropriate specialists"""
        recommendations = []
        
        # Check existing medical conditions
        for condition in user.medical_conditions:
            specialist_rec = cls._get_specialist_for_condition(condition)
            if specialist_rec:
                recommendations.append(specialist_rec)
        
        # Analyze symptoms
        symptom_recommendations = cls._analyze_symptoms(user)
        recommendations.extend(symptom_recommendations)
        
        # Analyze risk factors
        risk_recommendations = cls._analyze_risk_factors(user, metrics)
        recommendations.extend(risk_recommendations)
        
        # Age-based screening recommendations
        age_recommendations = cls._get_age_based_recommendations(user)
        recommendations.extend(age_recommendations)
        
        # Remove duplicates and prioritize
        recommendations = cls._prioritize_and_deduplicate(recommendations)
        
        return recommendations

    @classmethod
    def _get_specialist_for_condition(cls, condition: MedicalCondition) -> Optional[DoctorRecommendation]:
        """Get specialist recommendation for a medical condition"""
        if condition.specialist_required:
            return DoctorRecommendation(
                specialist_type=condition.specialist_required,
                urgency=condition.severity,
                reason=f"Management of {condition.name}",
                symptoms_indicators=[condition.name.lower()],
                timeline="within 2-4 weeks"
            )
        return None

    @classmethod
    def _analyze_symptoms(cls, user: UserProfile) -> List[DoctorRecommendation]:
        """Analyze user symptoms and recommend specialists"""
        recommendations = []
        
        # Check for specific symptoms
        if user.breathing_difficulties:
            recommendations.append(DoctorRecommendation(
                specialist_type=SpecialistType.PULMONOLOGIST,
                urgency=RiskLevel.MODERATE,
                reason="Breathing difficulties evaluation",
                symptoms_indicators=["breathing_difficulties", "shortness_of_breath"],
                recommended_tests=["Pulmonary Function Tests", "Chest X-ray"],
                timeline="within 2 weeks"
            ))
        
        if user.joint_pain:
            recommendations.append(DoctorRecommendation(
                specialist_type=SpecialistType.ORTHOPEDIST,
                urgency=RiskLevel.MODERATE,
                reason="Joint pain evaluation",
                symptoms_indicators=["joint_pain", "stiffness"],
                recommended_tests=["X-rays", "MRI if needed"],
                timeline="within 2-4 weeks"
            ))
        
        if user.mood_changes:
            recommendations.append(DoctorRecommendation(
                specialist_type=SpecialistType.PSYCHIATRIST,
                urgency=RiskLevel.MODERATE,
                reason="Mood changes evaluation",
                symptoms_indicators=["mood_changes", "depression", "anxiety"],
                timeline="within 2-4 weeks"
            ))
        
        return recommendations

    @classmethod
    def _analyze_risk_factors(cls, user: UserProfile, metrics: BodyMetrics) -> List[DoctorRecommendation]:
        """Analyze risk factors and recommend specialists"""
        recommendations = []
        
        # High BMI risk
        if metrics.bmi >= 30:
            recommendations.append(DoctorRecommendation(
                specialist_type=SpecialistType.ENDOCRINOLOGIST,
                urgency=RiskLevel.MODERATE,
                reason="Obesity and metabolic evaluation",
                symptoms_indicators=["obesity", "metabolic_syndrome"],
                recommended_tests=["Glucose Tolerance Test", "Lipid Profile", "HbA1c"],
                timeline="within 4 weeks"
            ))
        
        # High WHR risk
        whr_threshold = 0.90 if user.gender.upper() == 'M' else 0.85
        if metrics.whr > whr_threshold:
            recommendations.append(DoctorRecommendation(
                specialist_type=SpecialistType.CARDIOLOGIST,
                urgency=RiskLevel.MODERATE,
                reason="Cardiovascular risk assessment",
                symptoms_indicators=["abdominal_obesity", "cardiovascular_risk"],
                recommended_tests=["ECG", "Lipid Profile", "Stress Test"],
                timeline="within 4 weeks"
            ))
        
        return recommendations

    @classmethod
    def _get_age_based_recommendations(cls, user: UserProfile) -> List[DoctorRecommendation]:
        """Get age-based screening recommendations"""
        recommendations = []
        
        if user.age >= 50:
            # Bone density screening
            recommendations.append(DoctorRecommendation(
                specialist_type=SpecialistType.ORTHOPEDIST,
                urgency=RiskLevel.LOW,
                reason="Age-appropriate bone health screening",
                symptoms_indicators=["age_risk", "osteoporosis_risk"],
                recommended_tests=["Bone Density Scan"],
                timeline="within 6 months"
            ))
        
        if user.age >= 65:
            # Comprehensive geriatric evaluation
            recommendations.append(DoctorRecommendation(
                specialist_type=SpecialistType.CARDIOLOGIST,
                urgency=RiskLevel.MODERATE,
                reason="Age-appropriate cardiovascular screening",
                symptoms_indicators=["age_risk", "cardiovascular_risk"],
                recommended_tests=["ECG", "Echocardiogram", "Stress Test"],
                timeline="within 6 months"
            ))
        
        return recommendations

    @classmethod
    def _prioritize_and_deduplicate(cls, recommendations: List[DoctorRecommendation]) -> List[DoctorRecommendation]:
        """Remove duplicates and prioritize recommendations"""
        # Remove duplicates based on specialist type
        seen_specialists = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec.specialist_type not in seen_specialists:
                seen_specialists.add(rec.specialist_type)
                unique_recommendations.append(rec)
        
        # Sort by urgency
        unique_recommendations.sort(key=lambda x: cls._urgency_weight(x.urgency), reverse=True)
        
        return unique_recommendations

    @classmethod
    def _urgency_weight(cls, urgency: RiskLevel) -> int:
        """Convert urgency to numeric weight for sorting"""
        weights = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MODERATE: 2,
            RiskLevel.LOW: 1
        }
        return weights.get(urgency, 0)

class AdvancedHealthAdvisor:
    """Advanced health advisor with medical considerations"""

    def __init__(self):
        self.condition_manager = MedicalConditionManager()
        self.body_analyzer = AdvancedBodyAnalyzer()
        self.doctor_engine = DoctorRecommendationEngine()

    def _generate_mental_health_advice(self, user: UserProfile) -> List[AdviceItem]:
        """Generate mental health recommendations"""
        advice_items = []

        if user.stress_level >= 7:
            advice_items.append(AdviceItem(
                category="🧠 Mental Health",
                priority=RiskLevel.MODERATE,
                title="Elevated stress levels detected",
                description="Stress management strategies for better health",
                recommendations=[
                    "Practice mindfulness meditation 10-15 minutes daily",
                    "Consider cognitive behavioral therapy techniques",
                    "Maintain regular social connections",
                    "Ensure adequate sleep hygiene"
                ],
                follow_up_required=user.stress_level >= 9,
                timeline="ongoing"
            ))

        return advice_items

    def _generate_supplement_advice(self, metrics: BodyMetrics, user: UserProfile) -> List[AdviceItem]:
        """Generate personalized supplement recommendations"""
        advice_items = []
        recommendations = []

        # Vitamin D for low sunlight exposure or older adults
        if user.age > 50 or user.sleep_quality < 6:
            recommendations.append("Consider Vitamin D supplementation (1000-2000 IU daily)")

        # Add to existing advice generation
        if recommendations:
            advice_items.append(AdviceItem(
                category="💊 Supplement Guidance",
                priority=RiskLevel.LOW,
                title="Potential beneficial supplements",
                description="Always consult with your doctor before starting any supplements",
                recommendations=recommendations,
                medical_notes=["Check for medication interactions"],
                timeline="discuss with doctor"
            ))

        return advice_items

    def _validate_metrics(self, metrics: BodyMetrics) -> bool:
        """Validate body metrics are within reasonable ranges"""
        if not (10 < metrics.bmi < 50):
            logger.warning(f"Unrealistic BMI value: {metrics.bmi}")
            return False

        if not (0.5 < metrics.whr < 1.5):
            logger.warning(f"Unrealistic WHR value: {metrics.whr}")
            return False

        # Add more validation checks
        return True

    def generate_comprehensive_advice(self, metrics: BodyMetrics, user: UserProfile) -> List[AdviceItem]:
        """Generate comprehensive health advice with medical considerations"""
        if not self._validate_metrics(metrics):
            return [AdviceItem(
                category="⚠️ Data Validation",
                priority=RiskLevel.CRITICAL,
                title="Invalid health metrics detected",
                description="One or more body metrics appear unrealistic",
                recommendations=["Please verify your measurements and try again"],
                follow_up_required=True
            )]

        advice_items = []

        # Get doctor recommendations
        doctor_recommendations = self.doctor_engine.analyze_symptoms_and_recommend(user, metrics)
        
        # Medical screening first
        medical_restrictions = self._get_medical_restrictions(user)
        overall_risk = self.condition_manager.assess_overall_risk(user.medical_conditions)

        # Generate advice by category
        advice_items.extend(self._generate_weight_management_advice(metrics, user, medical_restrictions))
        advice_items.extend(self._generate_cardiovascular_advice(metrics, user, medical_restrictions))
        advice_items.extend(self._generate_exercise_advice(metrics, user, medical_restrictions))
        advice_items.extend(self._generate_nutrition_advice(metrics, user, medical_restrictions))
        advice_items.extend(self._generate_lifestyle_advice(metrics, user, medical_restrictions))
        advice_items.extend(self._generate_monitoring_advice(metrics, user, medical_restrictions))
        advice_items.extend(self._generate_mental_health_advice(user))
        advice_items.extend(self._generate_supplement_advice(metrics, user))

        # Add doctor recommendations to relevant advice items
        for item in advice_items:
            relevant_recommendations = [
                rec for rec in doctor_recommendations
                if any(symptom in item.title.lower() or symptom in item.description.lower()
                      for symptom in rec.symptoms_indicators)
            ]
            if relevant_recommendations:
                item.doctor_recommendations.extend(relevant_recommendations)

        # Sort by priority
        advice_items.sort(key=lambda x: self._priority_weight(x.priority), reverse=True)

        return advice_items

    def _get_medical_restrictions(self, user: UserProfile) -> Dict[str, List[str]]:
        """Compile all medical restrictions"""
        restrictions = {
            'exercise': [],
            'diet': [],
            'general': [],
            'monitoring': []
        }

        for condition in user.medical_conditions:
            restrictions['general'].extend(condition.restrictions)
            if condition.monitoring_required:
                restrictions['monitoring'].append(f"Monitor {condition.name}")

        # Pregnancy restrictions
        if user.pregnancy_status:
            restrictions['exercise'].extend([
                'avoid_supine_exercises_after_first_trimester',
                'avoid_contact_sports',
                'avoid_exercises_with_fall_risk',
                'limit_heart_rate_to_140bpm'
            ])
            restrictions['diet'].extend([
                'avoid_alcohol',
                'limit_caffeine',
                'ensure_folic_acid',
                'avoid_raw_foods'
            ])

        # Age-related restrictions
        if user.age >= 65:
            restrictions['exercise'].extend([
                'fall_prevention_focus',
                'gradual_progression',
                'balance_training_essential'
            ])

        return restrictions

    def _generate_weight_management_advice(self, metrics: BodyMetrics,
                                         user: UserProfile,
                                         restrictions: Dict) -> List[AdviceItem]:
        """Generate weight management advice"""
        advice_items = []

        # Check for eating disorder history
        has_eating_disorder = any('eating_disorder' in condition.name.lower()
                                for condition in user.medical_conditions)

        if metrics.bmi < 18.5:
            if has_eating_disorder:
                advice_items.append(AdviceItem(
                    category="⚠️ Weight Management - Special Considerations",
                    priority=RiskLevel.HIGH,
                    title="Underweight with eating disorder history",
                    description="Weight gain requires specialized medical supervision",
                    recommendations=[
                        "Work exclusively with eating disorder specialists",
                        "Focus on intuitive eating principles",
                        "Avoid calorie counting or restrictive approaches",
                        "Include mental health support",
                        "Consider family-based therapy if appropriate"
                    ],
                    restrictions=["no_calorie_counting", "no_weight_focus", "medical_supervision_required"],
                    medical_notes=["Requires eating disorder specialist", "Mental health monitoring essential"],
                    follow_up_required=True,
                    timeline="immediate"
                ))
            else:
                advice_items.append(self._create_healthy_weight_gain_advice(metrics, user))

        elif 25 <= metrics.bmi < 30:
            advice_items.append(self._create_weight_loss_advice(metrics, user, restrictions))

        elif metrics.bmi >= 30:
            advice_items.append(self._create_obesity_management_advice(metrics, user, restrictions))

        return advice_items

    def _create_healthy_weight_gain_advice(self, metrics: BodyMetrics, user: UserProfile) -> AdviceItem:
        """Create healthy weight gain advice"""
        recommendations = [
            "Aim for 0.5-1 lb weight gain per week",
            "Increase caloric intake by 300-500 calories daily",
            "Focus on nutrient-dense, calorie-rich foods",
            "Include protein with every meal (1.2-1.6g per kg body weight)",
            "Add healthy fats: nuts, avocados, olive oil",
            "Eat 5-6 smaller meals throughout the day"
        ]

        if user.age >= 65:
            recommendations.extend([
                "Consider protein supplements if appetite is poor",
                "Monitor for unintentional weight loss causes",
                "Ensure adequate vitamin D and calcium"
            ])

        return AdviceItem(
            category="💪 Healthy Weight Gain",
            priority=RiskLevel.MODERATE,
            title="Structured approach to healthy weight gain",
            description="Safe and sustainable weight gain strategy",
            recommendations=recommendations,
            medical_notes=["Consider underlying causes if rapid weight loss occurred"],
            follow_up_required=True,
            timeline="monthly monitoring"
        )

    def _create_weight_loss_advice(self, metrics: BodyMetrics, user: UserProfile,
                                 restrictions: Dict) -> AdviceItem:
        """Create weight loss advice with medical considerations"""

        # Check for diabetes
        has_diabetes = any('diabetes' in condition.name.lower()
                         for condition in user.medical_conditions)

        recommendations = [
            "Aim for 1-2 lbs weight loss per week",
            "Create moderate calorie deficit (300-500 calories)",
            "Prioritize protein to preserve muscle mass",
            "Include both cardio and strength training"
        ]

        restrictions_list = []
        medical_notes = []

        if has_diabetes:
            recommendations.extend([
                "Monitor blood glucose levels closely",
                "Coordinate meal timing with medications",
                "Choose low glycemic index foods",
                "Never skip meals"
            ])
            restrictions_list.extend(["no_extreme_calorie_restriction", "blood_glucose_monitoring"])
            medical_notes.append("Diabetes management coordination required")

        if user.age >= 50:
            recommendations.extend([
                "Focus on preserving muscle mass",
                "Include calcium and vitamin D rich foods",
                "Consider hormone level evaluation"
            ])

        return AdviceItem(
            category="🎯 Weight Management",
            priority=RiskLevel.MODERATE,
            title="Safe and effective weight loss strategy",
            description="Medically-supervised weight loss approach",
            recommendations=recommendations,
            restrictions=restrictions_list,
            medical_notes=medical_notes,
            follow_up_required=True,
            timeline="bi-weekly monitoring"
        )

    def _create_obesity_management_advice(self, metrics: BodyMetrics, user: UserProfile,
                                        restrictions: Dict) -> AdviceItem:
        """Create comprehensive obesity management advice"""

        metabolic_risk = self.body_analyzer.assess_metabolic_risk(metrics, user)

        recommendations = [
            "Comprehensive medical evaluation recommended",
            "Consider medically-supervised weight loss program",
            "Start with low-impact exercises",
            "Focus on sustainable lifestyle changes",
            "Consider working with registered dietitian"
        ]

        medical_notes = [
            "Screen for obesity-related comorbidities",
            "Consider metabolic syndrome evaluation",
            "Monitor for sleep apnea"
        ]

        if metabolic_risk == RiskLevel.HIGH:
            recommendations.extend([
                "Immediate medical consultation required",
                "Screen for diabetes and cardiovascular disease",
                "Consider pharmacological interventions"
            ])
            medical_notes.append("High metabolic syndrome risk - urgent medical attention")

        # Check for heart conditions
        has_heart_condition = any('heart' in condition.name.lower() or 'cardiovascular' in condition.name.lower()
                                for condition in user.medical_conditions)

        if has_heart_condition:
            recommendations.extend([
                "Cardiac rehabilitation program participation",
                "Heart rate monitoring during exercise",
                "Gradual exercise progression only"
            ])
            medical_notes.append("Cardiac clearance required for exercise")

        return AdviceItem(
            category="🚨 Comprehensive Obesity Management",
            priority=RiskLevel.HIGH,
            title="Medical-grade obesity intervention required",
            description="Comprehensive approach to obesity management with medical supervision",
            recommendations=recommendations,
            restrictions=["medical_supervision_required", "gradual_progression_only"],
            medical_notes=medical_notes,
            follow_up_required=True,
            timeline="immediate medical consultation"
        )

    def _generate_cardiovascular_advice(self, metrics: BodyMetrics, user: UserProfile,
                                      restrictions: Dict) -> List[AdviceItem]:
        """Generate cardiovascular health advice"""
        advice_items = []

        # High WHR assessment
        whr_threshold = 0.90 if user.gender.upper() == 'M' else 0.85
        if metrics.whr > whr_threshold:

            has_heart_disease = any('heart' in condition.name.lower() or 'cardiovascular' in condition.name.lower()
                                  for condition in user.medical_conditions)

            recommendations = [
                "Cardiovascular risk assessment recommended",
                "Focus on reducing abdominal fat",
                "Mediterranean-style diet implementation",
                "Regular cardiovascular exercise"
            ]

            restrictions_list = []
            medical_notes = []

            if has_heart_disease:
                recommendations.extend([
                    "Cardiac rehabilitation program enrollment",
                    "Heart rate monitoring during exercise",
                    "Stress testing before exercise program"
                ])
                restrictions_list.extend(["medical_clearance_required", "heart_rate_monitoring"])
                medical_notes.append("Cardiac clearance essential before exercise initiation")

            advice_items.append(AdviceItem(
                category="❤️ Cardiovascular Health",
                priority=RiskLevel.HIGH if has_heart_disease else RiskLevel.MODERATE,
                title="Elevated cardiovascular risk detected",
                description="Comprehensive cardiovascular risk management required",
                recommendations=recommendations,
                restrictions=restrictions_list,
                medical_notes=medical_notes,
                follow_up_required=True,
                timeline="within 2 weeks"
            ))

        return advice_items

    def _generate_exercise_advice(self, metrics: BodyMetrics, user: UserProfile,
                                restrictions: Dict) -> List[AdviceItem]:
        """Generate personalized exercise advice"""
        advice_items = []

        # Base exercise recommendations
        recommendations = []
        restrictions_list = restrictions.get('exercise', [])
        medical_notes = []

        # Age-based modifications
        if user.age >= 65:
            recommendations.extend([
                "Include balance training 2-3 times per week",
                "Focus on functional movements",
                "Start slowly and progress gradually",
                "Include flexibility exercises daily"
            ])
            medical_notes.append("Fall prevention focus essential for older adults")

        # Pregnancy modifications
        if user.pregnancy_status:
            recommendations = [
                "Maintain heart rate below 140 bpm",
                "Avoid supine positions after first trimester",
                "Focus on pelvic floor exercises",
                "Include prenatal yoga or swimming",
                "Stop exercise if experiencing dizziness or bleeding"
            ]
            restrictions_list.extend(["heart_rate_limitation", "position_restrictions"])
            medical_notes.append("Obstetric clearance recommended")

        # Medical condition modifications
        for condition in user.medical_conditions:
            if 'arthritis' in condition.name.lower():
                recommendations.extend([
                    "Focus on low-impact exercises (swimming, cycling)",
                    "Include gentle range-of-motion exercises",
                    "Apply heat before exercise, ice after if swollen"
                ])

            elif 'osteoporosis' in condition.name.lower():
                recommendations.extend([
                    "Weight-bearing exercises essential",
                    "Avoid forward spinal flexion",
                    "Include balance training"
                ])
                restrictions_list.append("no_spinal_flexion")

        # Activity level considerations
        if user.activity_level == ActivityLevel.SEDENTARY:
            recommendations.extend([
                "Start with 10-minute exercise sessions",
                "Focus on daily movement first",
                "Gradually increase to 150 minutes per week"
            ])

        if recommendations:  # Only add if we have specific recommendations
            advice_items.append(AdviceItem(
                category="🏃‍♂️ Exercise Program",
                priority=RiskLevel.MODERATE,
                title="Personalized exercise recommendations",
                description="Safe and effective exercise program tailored to your needs",
                recommendations=recommendations,
                restrictions=restrictions_list,
                medical_notes=medical_notes,
                follow_up_required=False,
                timeline="ongoing"
            ))

        return advice_items

    def _generate_nutrition_advice(self, metrics: BodyMetrics, user: UserProfile,
                                 restrictions: Dict) -> List[AdviceItem]:
        """Generate personalized nutrition advice"""
        advice_items = []

        recommendations = []
        restrictions_list = restrictions.get('diet', [])
        medical_notes = []

        # Diabetes-specific nutrition
        has_diabetes = any('diabetes' in condition.name.lower()
                         for condition in user.medical_conditions)

        if has_diabetes:
            recommendations.extend([
                "Follow consistent carbohydrate counting",
                "Choose low glycemic index foods",
                "Include fiber-rich foods at every meal",
                "Never skip meals",
                "Monitor blood glucose response to foods"
            ])
            medical_notes.append("Coordinate with diabetes educator")

        # Hypertension nutrition
        has_hypertension = any('hypertension' in condition.name.lower()
                             for condition in user.medical_conditions)

        if has_hypertension:
            recommendations.extend([
                "Follow DASH diet principles",
                "Limit sodium to 2300mg daily (ideally 1500mg)",
                "Increase potassium-rich foods",
                "Limit alcohol consumption"
            ])
            restrictions_list.append("sodium_restriction")
            medical_notes.append("Monitor blood pressure response to dietary changes")

        # Kidney disease nutrition
        has_kidney_disease = any('kidney' in condition.name.lower()
                               for condition in user.medical_conditions)

        if has_kidney_disease:
            recommendations.extend([
                "Work with renal dietitian",
                "Monitor protein intake as directed",
                "Limit potassium and phosphorus as needed",
                "Control fluid intake if required"
            ])
            restrictions_list.extend(["protein_restriction", "potassium_restriction"])
            medical_notes.append("Renal dietitian consultation essential")

        # General healthy recommendations
        if not any([has_diabetes, has_hypertension, has_kidney_disease]):
            recommendations.extend([
                "Follow Mediterranean-style eating pattern",
                "Include 5-9 servings of fruits and vegetables daily",
                "Choose whole grains over refined grains",
                "Include lean protein sources",
                "Stay hydrated with 8-10 glasses of water daily"
            ])

        if recommendations:
            advice_items.append(AdviceItem(
                category="🥗 Nutrition Strategy",
                priority=RiskLevel.MODERATE,
                title="Personalized nutrition recommendations",
                description="Medically-appropriate nutrition plan",
                recommendations=recommendations,
                restrictions=restrictions_list,
                medical_notes=medical_notes,
                follow_up_required=bool(medical_notes),
                timeline="ongoing with periodic review"
            ))

        return advice_items

    def _generate_lifestyle_advice(self, metrics: BodyMetrics, user: UserProfile,
                                 restrictions: Dict) -> List[AdviceItem]:
        """Generate lifestyle modification advice"""
        advice_items = []

        recommendations = []
        medical_notes = []

        # Sleep optimization
        if user.sleep_quality < 6:
            recommendations.extend([
                "Establish consistent sleep schedule",
                "Create sleep-conducive environment",
                "Limit screen time 1 hour before bed",
                "Consider sleep study if snoring/breathing issues"
            ])
            if metrics.bmi >= 30:
                medical_notes.append("Screen for sleep apnea")

        # Stress management
        if user.stress_level >= 7:
            recommendations.extend([
                "Practice daily stress reduction techniques",
                "Consider meditation or mindfulness training",
                "Evaluate work-life balance",
                "Consider counseling if needed"
            ])

        # Smoking cessation
        if user.smoking_status:
            recommendations.extend([
                "Smoking cessation program enrollment strongly recommended",
                "Consider nicotine replacement therapy",
                "Identify smoking triggers and alternatives"
            ])
            medical_notes.append("Smoking cessation significantly improves all health outcomes")

        # Alcohol management
        if user.alcohol_consumption in ['moderate', 'heavy']:
            recommendations.extend([
                "Evaluate alcohol consumption patterns",
                "Consider reducing to recommended limits",
                "Discuss alcohol use with healthcare provider"
            ])
            if user.alcohol_consumption == 'heavy':
                medical_notes.append("Alcohol reduction program may be beneficial")

        if recommendations:
            advice_items.append(AdviceItem(
                category="🌱 Lifestyle Optimization",
                priority=RiskLevel.MODERATE,
                title="Comprehensive lifestyle modifications",
                description="Evidence-based lifestyle improvements for better health",
                recommendations=recommendations,
                medical_notes=medical_notes,
                follow_up_required=bool(medical_notes),
                timeline="ongoing"
            ))

        return advice_items

    def _generate_monitoring_advice(self, metrics: BodyMetrics, user: UserProfile,
                                  restrictions: Dict) -> List[AdviceItem]:
        """Generate monitoring and follow-up advice"""
        advice_items = []

        monitoring_items = []
        follow_up_timeline = []

        # Condition-specific monitoring
        monitoring_required_conditions = [c for c in user.medical_conditions if c.monitoring_required]

        for condition in monitoring_required_conditions:
            if 'diabetes' in condition.name.lower():
                monitoring_items.extend([
                    "Daily blood glucose monitoring",
                    "HbA1c every 3-6 months",
                    "Annual eye and foot exams"
                ])
                follow_up_timeline.append("Diabetes: Every 3 months")

            elif 'hypertension' in condition.name.lower():
                monitoring_items.extend([
                    "Daily blood pressure monitoring",
                    "Regular medication compliance check"
                ])
                follow_up_timeline.append("Hypertension: Every 2-3 months")

            elif 'heart' in condition.name.lower():
                monitoring_items.extend([
                    "Regular ECG monitoring",
                    "Lipid profile every 6 months",
                    "Exercise tolerance assessment"
                ])
                follow_up_timeline.append("Cardiovascular: Every 3-6 months")

        # General monitoring based on risk factors
        metabolic_risk = self.body_analyzer.assess_metabolic_risk(metrics, user)

        if metabolic_risk in [RiskLevel.MODERATE, RiskLevel.HIGH]:
            monitoring_items.extend([
                "Monthly weight and waist measurements",
                "Quarterly blood pressure checks",
                "Annual lipid panel and glucose screening"
            ])

        # Age-based monitoring
        if user.age >= 50:
            monitoring_items.extend([
                "Annual comprehensive physical exam",
                "Bone density screening (if appropriate)",
                "Cancer screening as per guidelines"
            ])

        if monitoring_items:
            advice_items.append(AdviceItem(
                category="📊 Health Monitoring Plan",
                priority=RiskLevel.HIGH,
                title="Comprehensive health monitoring strategy",
                description="Regular monitoring to track progress and identify issues early",
                recommendations=monitoring_items,
                medical_notes=follow_up_timeline,
                follow_up_required=True,
                timeline="as specified per recommendation"
            ))

        return advice_items

    def _priority_weight(self, priority: RiskLevel) -> int:
        """Convert priority to numeric weight for sorting"""
        weights = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MODERATE: 2,
            RiskLevel.LOW: 1
        }
        return weights.get(priority, 0)

class HealthAdvisorFormatter:
    """Format and display health advice"""

    @staticmethod
    def format_advice_for_display(advice_items: List[AdviceItem]) -> str:
        """Format advice items for user-friendly display"""

        output = []
        output.append("=" * 80)
        output.append("🏥 ADVANCED MEDICAL HEALTH ASSESSMENT REPORT")
        output.append("=" * 80)
        output.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")

        # Group by priority
        critical_items = [item for item in advice_items if item.priority == RiskLevel.CRITICAL]
        high_items = [item for item in advice_items if item.priority == RiskLevel.HIGH]
        moderate_items = [item for item in advice_items if item.priority == RiskLevel.MODERATE]
        low_items = [item for item in advice_items if item.priority == RiskLevel.LOW]

        if critical_items:
            output.append("🚨 CRITICAL PRIORITY - IMMEDIATE MEDICAL ATTENTION REQUIRED")
            output.append("-" * 60)
            for item in critical_items:
                output.extend(HealthAdvisorFormatter._format_single_advice(item))
            output.append("")

        if high_items:
            output.append("⚠️ HIGH PRIORITY - MEDICAL CONSULTATION RECOMMENDED")
            output.append("-" * 60)
            for item in high_items:
                output.extend(HealthAdvisorFormatter._format_single_advice(item))
            output.append("")

        if moderate_items:
            output.append("📋 MODERATE PRIORITY - LIFESTYLE MODIFICATIONS")
            output.append("-" * 60)
            for item in moderate_items:
                output.extend(HealthAdvisorFormatter._format_single_advice(item))
            output.append("")

        if low_items:
            output.append("💡 GENERAL WELLNESS RECOMMENDATIONS")
            output.append("-" * 60)
            for item in low_items:
                output.extend(HealthAdvisorFormatter._format_single_advice(item))
            output.append("")

        # Add disclaimer
        output.append("=" * 80)
        output.append("⚠️ MEDICAL DISCLAIMER")
        output.append("=" * 80)
        output.append("This assessment is for informational purposes only and does not")
        output.append("constitute medical advice. Always consult with qualified healthcare")
        output.append("professionals before making significant changes to your diet,")
        output.append("exercise routine, or medical management.")
        output.append("")
        output.append("If you experience any concerning symptoms, seek immediate")
        output.append("medical attention.")
        output.append("=" * 80)

        return "\n".join(output)

    @staticmethod
    def _format_single_advice(item: AdviceItem) -> List[str]:
        """Format a single advice item"""
        lines = []

        lines.append(f"{item.category}")
        lines.append(f"Title: {item.title}")
        lines.append(f"Timeline: {item.timeline}")

        if item.description:
            lines.append(f"Description: {item.description}")

        lines.append("Recommendations:")
        for rec in item.recommendations:
            lines.append(f" • {rec}")

        if item.restrictions:
            lines.append("⚠️ Important Restrictions:")
            for restriction in item.restrictions:
                lines.append(f" ⚠️ {restriction.replace('_', ' ').title()}")

        if item.medical_notes:
            lines.append("🩺 Medical Notes:")
            for note in item.medical_notes:
                lines.append(f" 🩺 {note}")

        if item.follow_up_required:
            lines.append("📅 Follow-up Required: Yes")

        lines.append("-" * 40)
        return lines

    @staticmethod
    def format_for_html(advice_items: List[AdviceItem]) -> str:
        """Format advice for web display with HTML"""
        html_output = []
        html_output.append("<div class='health-report'>")
        html_output.append("<h1>Advanced Health Assessment Report</h1>")

        for item in sorted(advice_items, key=lambda x: x.priority.value, reverse=True):
            priority_class = item.priority.name.lower()
            html_output.append(f"<div class='advice-item {priority_class}'>")
            html_output.append(f"<h2>{item.title}</h2>")
            html_output.append(f"<p class='priority'>{item.priority.value.upper()} PRIORITY</p>")
            html_output.append(f"<p>{item.description}</p>")

            if item.recommendations:
                html_output.append("<h3>Recommendations:</h3><ul>")
                for rec in item.recommendations:
                    html_output.append(f"<li>{rec}</li>")
                html_output.append("</ul>")

            html_output.append("</div>")

        html_output.append("</div>")
        return "\n".join(html_output)

def generate_health_advice(bmi: float, whr: float, whtr: float, bfp: float,
                         waist_cm: float, gender: str, arm_to_leg_ratio: float) -> List[Dict]:
    """Legacy function to maintain compatibility with existing code"""
    try:
        # Create basic user profile
        user_profile = UserProfile(
            age=45,  # Default age
            gender=gender,
            activity_level=ActivityLevel.MODERATELY_ACTIVE
        )

        # Create body metrics
        body_metrics = BodyMetrics(
            bmi=bmi,
            whr=whr,
            whtr=whtr,
            body_fat_percentage=bfp,
            muscle_mass_percentage=100 - bfp - 15,  # Simplified calculation
            waist_cm=waist_cm,
            arm_to_leg_ratio=arm_to_leg_ratio
        )

        # Generate advice using advanced system
        advisor = AdvancedHealthAdvisor()
        advice_items = advisor.generate_comprehensive_advice(body_metrics, user_profile)

        # Convert to legacy format
        legacy_advice = []
        for item in advice_items:
            legacy_advice.append({
                "category": item.category,
                "title": item.title,
                "suggestions": item.recommendations
            })

        return legacy_advice

    except Exception as e:
        logger.error(f"Error generating health advice: {e}")
        return [{
            "category": "❌ Error",
            "title": "Unable to generate health advice",
            "suggestions": [f"An error occurred: {str(e)}"]
        }]

def print_health_advice(advice_list: List[Dict]):
    """Print the health advice in a formatted way"""
    try:
        # Convert legacy format to new format
        advice_items = []
        for advice in advice_list:
            advice_items.append(AdviceItem(
                category=advice['category'],
                priority=RiskLevel.MODERATE,  # Default priority
                title=advice['title'],
                description="",
                recommendations=advice['suggestions']
            ))

        # Use new formatter
        formatted_output = HealthAdvisorFormatter.format_advice_for_display(advice_items)
        print(formatted_output)

    except Exception as e:
        logger.error(f"Error printing health advice: {e}")
        print(f"❌ Error printing health advice: {str(e)}")

def generate_advice_from_biometrics(biometric_data: Dict) -> List[AdviceItem]:
    """Generate advice from wearable device data"""
    advisor = AdvancedHealthAdvisor()

    # Create metrics from wearable data
    metrics = BodyMetrics(
        bmi=biometric_data.get('bmi'),
        whr=biometric_data.get('whr'),
        whtr=biometric_data.get('whtr'),
        body_fat_percentage=biometric_data.get('body_fat'),
        waist_cm=biometric_data.get('waist_circumference'),
        muscle_mass_percentage=biometric_data.get('muscle_mass', 0),
        arm_to_leg_ratio=biometric_data.get('arm_to_leg_ratio', 1.0),
        resting_heart_rate=biometric_data.get('resting_hr'),
        blood_pressure_systolic=biometric_data.get('bp_systolic'),
        blood_pressure_diastolic=biometric_data.get('bp_diastolic')
    )

    # Create user profile
    profile = UserProfile(
        age=biometric_data.get('age'),
        gender=biometric_data.get('gender'),
        activity_level=ActivityLevel(biometric_data.get('activity_level', 'moderately_active')),
        family_history=biometric_data.get('family_history', {}),
        recent_bloodwork=biometric_data.get('bloodwork', {}),
        fitness_goals=biometric_data.get('fitness_goals', []),
        dietary_preferences=biometric_data.get('dietary_preferences', []),
        current_symptoms=biometric_data.get('symptoms', []),
        pain_locations=biometric_data.get('pain_locations', []),
        chronic_fatigue=biometric_data.get('chronic_fatigue', False),
        frequent_headaches=biometric_data.get('frequent_headaches', False),
        digestive_issues=biometric_data.get('digestive_issues', False),
        breathing_difficulties=biometric_data.get('breathing_difficulties', False),
        skin_problems=biometric_data.get('skin_problems', False),
        vision_problems=biometric_data.get('vision_problems', False),
        hearing_problems=biometric_data.get('hearing_problems', False),
        joint_pain=biometric_data.get('joint_pain', False),
        muscle_weakness=biometric_data.get('muscle_weakness', False),
        mood_changes=biometric_data.get('mood_changes', False)
    )

    return advisor.generate_comprehensive_advice(metrics, profile)

def export_advice_to_json(advice_items: List[AdviceItem]) -> str:
    """Export advice to JSON format for API responses"""
    return json.dumps([
        {
            "category": item.category,
            "priority": item.priority.value,
            "title": item.title,
            "recommendations": item.recommendations,
            "follow_up_required": item.follow_up_required
        }
        for item in advice_items
    ], indent=2)

if __name__ == "__main__":
    # Example usage
    from BMI_calc import get_body_composition_summary, load_current_user

    try:
        current_user_id = load_current_user()
        if current_user_id is None:
            print("❌ Please log in first to get health advice.")
            print("Run 'python Login_page.py' to log in.")
        else:
            # Get user data and calculate metrics
            from BMI_calc import get_user_data, calculate_age
            user_data = get_user_data(current_user_id)

            # Calculate age
            age = calculate_age(user_data['date_of_birth'])

            # Get measurements
            weight_kg = user_data['weight']
            height_cm = user_data['height']
            height_m = height_cm / 100
            gender = user_data['gender']
            waist_cm = user_data['waist']
            hip_cm = user_data['hip']
            arm_length_cm = user_data['arm_length']
            leg_length_cm = user_data['leg_length']

            # Calculate metrics
            bmi = weight_kg / (height_m ** 2)
            whr = waist_cm / hip_cm
            whtr = waist_cm / height_cm
            bfp = (1.20 * bmi) + (0.23 * age) - (16.2 if gender.upper() == 'M' else 5.4)
            arm_to_leg_ratio = arm_length_cm / leg_length_cm

            # Generate and print advice
            advice = generate_health_advice(bmi, whr, whtr, bfp, waist_cm, gender, arm_to_leg_ratio)
            print_health_advice(advice)

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}") 