import re
from collections import Counter
import numpy as np
from urllib.parse import urlparse
import emoji
from sklearn.ensemble import RandomForestClassifier

def extract_text_features(comment):
    """
    Extract text-based features from a YouTube comment.
    
    Parameters:
    comment (str): The comment text to analyze
    
    Returns:
    dict: Dictionary containing text features
    """
    # Convert to lowercase for some analyses
    lower_comment = comment.lower()
    
    # Basic length features
    features = {
        'length': len(comment),
        'word_count': len(comment.split()),
        'avg_word_length': np.mean([len(word) for word in comment.split()]) if comment else 0,
    }
    
    # Character ratios
    features.update({
        'caps_ratio': sum(1 for c in comment if c.isupper()) / (len(comment) if len(comment) > 0 else 1),
        'digits_ratio': sum(1 for c in comment if c.isdigit()) / (len(comment) if len(comment) > 0 else 1),
        'special_chars_ratio': sum(1 for c in comment if not c.isalnum() and not c.isspace()) / (len(comment) if len(comment) > 0 else 1),
    })
    
    # URL features
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', comment)
    features.update({
        'contains_url': int(len(urls) > 0),
        'url_count': len(urls),
    })
    
    # Repetition features
    words = comment.split()
    word_counts = Counter(words)
    features.update({
        'unique_words_ratio': len(set(words)) / (len(words) if len(words) > 0 else 1),
        'max_word_repetition': max(word_counts.values()) if word_counts else 0,
    })
    
    # Character repetition
    char_repetition = re.findall(r'(.)\1{2,}', comment)
    features['repeated_chars_count'] = len(char_repetition)
    
    # Emoji features
    emoji_list = [c for c in comment if c in emoji.EMOJI_DATA]
    features.update({
        'emoji_count': len(emoji_list),
        'emoji_ratio': len(emoji_list) / (len(comment) if len(comment) > 0 else 1),
    })
    
    # Punctuation features
    exclamation_count = comment.count('!')
    question_count = comment.count('?')
    features.update({
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'punctuation_ratio': (exclamation_count + question_count) / (len(comment) if len(comment) > 0 else 1),
    })
    
    # Common spam patterns
    spam_patterns = [
        r'subscribe|sub4sub|follow|check out|visit|click|win|free|prize|money|lottery|discount|offer',
        r'!!+|\?\?+',  # Multiple exclamation or question marks
        r'www\.|\.com|\.net|\.org',  # Website patterns
        r'whatsapp|telegram|insta[gram]?',  # Social media references
    ]
    
    for i, pattern in enumerate(spam_patterns):
        features[f'spam_pattern_{i}'] = int(bool(re.search(pattern, lower_comment)))
    
    # Hashtag and mention features
    features.update({
        'hashtag_count': len(re.findall(r'#\w+', comment)),
        'mention_count': len(re.findall(r'@\w+', comment)),
    })
    
    return features

def create_feature_matrix(comments):
    """
    Create a feature matrix from a list of comments.
    
    Parameters:
    comments (list): List of comment strings
    
    Returns:
    tuple: (feature_matrix, feature_names)
    """
    # Extract features for all comments
    feature_dicts = [extract_text_features(comment) for comment in comments]
    
    # Get feature names from the first comment
    feature_names = list(feature_dicts[0].keys())
    
    # Create feature matrix
    feature_matrix = np.array([[features[fname] for fname in feature_names] 
                             for features in feature_dicts])
    
    return feature_matrix, feature_names