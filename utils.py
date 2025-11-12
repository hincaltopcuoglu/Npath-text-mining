"""
Yardımcı fonksiyonlar: Text preprocessing, tokenization, cleaning
"""
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from typing import List, Tuple
import pandas as pd

# NLTK data indirme (ilk çalıştırmada gerekli)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Text verilerini işleme ve sequence'lere çevirme"""
    
    def __init__(self, 
                 remove_stopwords: bool = False,
                 lowercase: bool = True,
                 min_word_length: int = 2,
                 language: str = 'turkish'):
        """
        Args:
            remove_stopwords: Stopword'leri kaldır
            lowercase: Küçük harfe çevir
            min_word_length: Minimum kelime uzunluğu
            language: Dil (turkish/english)
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.min_word_length = min_word_length
        
        if language == 'turkish':
            # Türkçe stopwords (basit liste, gerekirse genişletilebilir)
            self.stopwords = set(['ve', 'ile', 'bir', 'bu', 'şu', 'o', 'de', 'da', 'ki', 'mi', 'mu', 'mü'])
        else:
            self.stopwords = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Text temizleme"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Küçük harfe çevir
        if self.lowercase:
            text = text.lower()
        
        # Özel karakterleri temizle (noktalama işaretlerini koru)
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_sentence(self, text: str) -> List[str]:
        """Cümleyi kelimelere ayır"""
        text = self.clean_text(text)
        
        # Cümle tokenization
        sentences = sent_tokenize(text)
        
        all_tokens = []
        for sentence in sentences:
            # Kelime tokenization
            tokens = word_tokenize(sentence)
            
            # Filtreleme
            filtered_tokens = []
            for token in tokens:
                # Noktalama işaretlerini atla
                if token in ['.', ',', '!', '?', ';', ':']:
                    continue
                
                # Minimum uzunluk kontrolü
                if len(token) < self.min_word_length:
                    continue
                
                # Stopword kontrolü
                if self.remove_stopwords and token in self.stopwords:
                    continue
                
                filtered_tokens.append(token)
            
            if filtered_tokens:
                all_tokens.extend(filtered_tokens)
        
        return all_tokens
    
    def extract_sequences(self, text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """
        Text'ten n-gram sequence'leri çıkar
        
        Args:
            text: İşlenecek text
            n: n-gram boyutu (2 = bigram, 3 = trigram, vb.)
        
        Returns:
            List of tuples: [(word1, word2), (word2, word3), ...]
        """
        tokens = self.tokenize_sentence(text)
        
        if len(tokens) < n:
            return []
        
        sequences = []
        for i in range(len(tokens) - n + 1):
            sequence = tuple(tokens[i:i+n])
            sequences.append(sequence)
        
        return sequences
    
    def extract_word_sequences(self, text: str) -> List[str]:
        """Text'ten kelime dizisini çıkar (path için)"""
        return self.tokenize_sentence(text)
    
    def process_dataframe(self, 
                         df: pd.DataFrame, 
                         text_column: str,
                         category_column: str) -> pd.DataFrame:
        """
        DataFrame'i işle ve sequence'leri ekle
        
        Returns:
            DataFrame with 'tokens' and 'sequences' columns
        """
        processed_df = df.copy()
        
        # Tokenization
        processed_df['tokens'] = processed_df[text_column].apply(
            self.extract_word_sequences
        )
        
        # Bigram sequences (2-word paths)
        processed_df['bigrams'] = processed_df[text_column].apply(
            lambda x: self.extract_sequences(x, n=2)
        )
        
        # Trigram sequences (3-word paths)
        processed_df['trigrams'] = processed_df[text_column].apply(
            lambda x: self.extract_sequences(x, n=3)
        )
        
        return processed_df

