"""
Feature Name Translator Service
Converts technical ML feature names and SHAP values into salesperson-friendly language.
"""

from typing import Dict, List, Any
import re


class ExplanationTranslator:
    """Translates technical feature names to plain English explanations."""
    
    def __init__(self):
        # Feature name patterns and their translations
        self.feature_patterns = {
            # Engagement metrics
            r"engagement.*score": {
                "positive": "Strong recent engagement",
                "negative": "Low engagement history",
                "neutral": "Moderate engagement"
            },
            r"email.*open": {
                "positive": "Actively opens emails",
                "negative": "Rarely opens emails",
                "neutral": "Occasional email opens"
            },
            r"email.*click": {
                "positive": "Clicks through emails frequently",
                "negative": "Doesn't click email links",
                "neutral": "Some email clicks"
            },
            r"reply.*count|replied.*times": {
                "positive": "Replied {value} times recently",
                "negative": "No recent replies",
                "neutral": "Replied {value} times"
            },
            r"last.*interaction|recent.*activity": {
                "positive": "Very recent activity",
                "negative": "Inactive for a while",
                "neutral": "Some recent activity"
            },
            
            # Deal/Pipeline metrics
            r"deal.*age|days.*in.*pipeline": {
                "positive": "Fresh opportunity",
                "negative": "Deal going cold — act fast",
                "neutral": "Standard deal timeline"
            },
            r"deal.*value|deal.*size|revenue": {
                "positive": "High-value opportunity",
                "negative": "Small deal size",
                "neutral": "Mid-size deal"
            },
            r"stage|pipeline.*stage": {
                "positive": "Advanced in sales process",
                "negative": "Early stage",
                "neutral": "Mid-stage opportunity"
            },
            
            # Company/Firmographic
            r"company.*size|employee.*count|headcount": {
                "positive": "Perfect company size",
                "negative": "Outside target size",
                "neutral": "Acceptable company size"
            },
            r"industry|vertical|sector": {
                "positive": "Ideal industry fit",
                "negative": "Industry mismatch",
                "neutral": "Industry fit"
            },
            r"revenue|arr|annual.*revenue": {
                "positive": "Strong revenue profile",
                "negative": "Below revenue threshold",
                "neutral": "Moderate revenue"
            },
            r"location|region|geography|country": {
                "positive": "Target market location",
                "negative": "Outside target region",
                "neutral": "Location fit"
            },
            
            # Behavioral/Intent
            r"website.*visit|page.*view": {
                "positive": "Actively browsing website",
                "negative": "No website visits",
                "neutral": "Visited website"
            },
            r"demo.*request|trial.*signup": {
                "positive": "Requested demo/trial",
                "negative": "No demo interest shown",
                "neutral": "Demo interest"
            },
            r"download|content.*download": {
                "positive": "Downloaded resources",
                "negative": "No content downloads",
                "neutral": "Some downloads"
            },
            r"meeting.*scheduled|calendar.*invite": {
                "positive": "Meeting scheduled",
                "negative": "Declined meetings",
                "neutral": "Meeting proposed"
            },
            
            # Contact Quality
            r"title|job.*title|role": {
                "positive": "Decision-maker role",
                "negative": "Not a decision-maker",
                "neutral": "Relevant role"
            },
            r"seniority|level": {
                "positive": "Senior-level contact",
                "negative": "Junior contact",
                "neutral": "Mid-level contact"
            },
            r"budget|authority": {
                "positive": "Has budget authority",
                "negative": "No budget authority",
                "neutral": "Some budget influence"
            },
            
            # Previous relationship
            r"past.*customer|previous.*purchase|churned": {
                "positive": "Former customer — win-back opportunity",
                "negative": "Previously churned",
                "neutral": "Past customer"
            },
            r"referral|referred.*by": {
                "positive": "Came from referral",
                "negative": "Cold outreach",
                "neutral": "Referral connection"
            },
            
            # Timing/Urgency
            r"contract.*expir|renewal.*date": {
                "positive": "Contract expiring soon",
                "negative": "Just renewed elsewhere",
                "neutral": "Contract timing"
            },
            r"intent.*signal|buying.*signal": {
                "positive": "Strong buying signals",
                "negative": "No buying signals",
                "neutral": "Some interest signals"
            },
            
            # Generic fallbacks
            r"score$": {
                "positive": "High score",
                "negative": "Low score",
                "neutral": "Score factor"
            },
            r"count$": {
                "positive": "High activity count",
                "negative": "Low activity count",
                "neutral": "Activity count"
            }
        }
    
    def translate_feature(self, feature_name: str, contribution: float, value: Any = None) -> str:
        """Translate a single feature to plain English."""
        feature_lower = feature_name.lower()
        
        # Determine sentiment
        if contribution > 0.05:
            sentiment = "positive"
        elif contribution < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Match against patterns
        for pattern, translations in self.feature_patterns.items():
            if re.search(pattern, feature_lower):
                translation = translations.get(sentiment, translations.get("neutral"))
                
                # Template substitution if value provided
                if value is not None and "{value}" in translation:
                    translation = translation.replace("{value}", str(value))
                
                return translation
        
        # Fallback: prettify the feature name
        return self._prettify_feature_name(feature_name, sentiment)
    
    def _prettify_feature_name(self, feature_name: str, sentiment: str) -> str:
        """Convert snake_case or camelCase to readable text."""
        cleaned = re.sub(r'^(is_|has_|num_|count_|total_)', '', feature_name)
        cleaned = re.sub(r'(_score|_count|_rate)$', '', cleaned)
        
        words = re.split(r'[_\s]+', cleaned)
        readable = ' '.join(word.capitalize() for word in words)
        
        if sentiment == "positive":
            return f"Good {readable}"
        elif sentiment == "negative":
            return f"Low {readable}"
        else:
            return readable
    
    def translate_rationale(
        self,
        top_positive: List[Dict[str, Any]],
        top_negative: List[Dict[str, Any]],
        max_items: int = 3
    ) -> Dict[str, Any]:
        """Translate SHAP rationale into plain English."""
        positive_drivers = []
        negative_drivers = []
        
        # Translate positive drivers
        for item in top_positive[:max_items]:
            label = item.get('label', '')
            contribution = item.get('contribution', 0)
            value = item.get('value')
            
            translated = self.translate_feature(label, contribution, value)
            positive_drivers.append({
                "text": translated,
                "original": label,
                "contribution": contribution,
                "sentiment": "positive"
            })
        
        # Translate negative drivers
        for item in top_negative[:max_items]:
            label = item.get('label', '')
            contribution = item.get('contribution', 0)
            value = item.get('value')
            
            translated = self.translate_feature(label, contribution, value)
            negative_drivers.append({
                "text": translated,
                "original": label,
                "contribution": contribution,
                "sentiment": "negative"
            })
        
        # Generate summary sentence
        summary = self._generate_summary(positive_drivers, negative_drivers)
        
        return {
            "positive_drivers": positive_drivers,
            "negative_drivers": negative_drivers,
            "summary": summary,
            "all_drivers": positive_drivers + negative_drivers
        }
    
    def _generate_summary(self, positive: List[Dict], negative: List[Dict]) -> str:
        """Generate a conversational summary sentence."""
        if not positive and not negative:
            return "Score based on overall profile"
        
        if positive and not negative:
            driver_text = ", ".join([d["text"] for d in positive[:2]])
            return f"Strong candidate because of {driver_text}"
        
        if negative and not positive:
            driver_text = ", ".join([d["text"] for d in negative[:2]])
            return f"Lower score due to {driver_text}"
        
        pos_text = positive[0]["text"] if positive else ""
        neg_text = negative[0]["text"] if negative else ""
        
        return f"Strong on {pos_text}, but {neg_text}"
    
    def recommend_action(self, score: float, drivers: List[Dict[str, Any]]) -> str:
        """Recommend best action based on score and drivers."""
        driver_texts = " ".join([d.get("text", "").lower() for d in drivers])
        
        # High urgency signals
        if any(keyword in driver_texts for keyword in ["going cold", "act fast", "expiring", "scheduled"]):
            return "call"
        
        # High score logic
        if score >= 80:
            if any(keyword in driver_texts for keyword in ["engagement", "replied", "active", "demo"]):
                return "call"
            return "email"
        
        # Medium score
        if score >= 55:
            return "email"
        
        # Low score
        return "nurture"
    
    def get_action_priority(self, score: float) -> str:
        """Map score to action priority."""
        if score >= 80:
            return "high"
        elif score >= 55:
            return "medium"
        else:
            return "low"
    
    def enrich_scoring_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a scoring result with sales-friendly fields."""
        score = result.get("score", 0)
        rationale = result.get("rationale", {})
        
        top_positive = rationale.get("top_positive", [])
        top_negative = rationale.get("top_negative", [])
        
        # Translate rationale
        translated = self.translate_rationale(top_positive, top_negative)
        
        # Get recommendations
        action = self.recommend_action(score, translated["all_drivers"])
        priority = self.get_action_priority(score)
        
        # Determine score band
        if score >= 80:
            band = "hot"
            band_label = "Hot Lead"
        elif score >= 55:
            band = "warm"
            band_label = "Warm Lead"
        else:
            band = "cold"
            band_label = "Cold Lead"
        
        # Add enriched fields - SHOW PERCENTAGE
        result["display_score"] = round(score, 1)
        result["score_percentage"] = f"{round(score)}%"
        result["action_priority"] = priority
        result["recommended_action"] = action
        result["score_band"] = band
        result["score_band_label"] = band_label
        result["plain_english"] = translated
        
        return result


# Singleton instance
_translator = ExplanationTranslator()


def translate_scoring_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translate a batch of scoring results."""
    return [_translator.enrich_scoring_result(result) for result in results]


def get_translator() -> ExplanationTranslator:
    """Get the singleton translator instance."""
    return _translator
