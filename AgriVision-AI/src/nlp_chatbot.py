"""TF-IDF based agricultural chatbot with dataset-aware answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NLPChatbot:
    """Simple NLP-based chatbot for agricultural queries"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        self.qa_pairs = self._create_qa_pairs()
        self.questions = [qa['question'] for qa in self.qa_pairs]
        self.answers = [qa['answer'] for qa in self.qa_pairs]
        
        # Fit vectorizer on questions
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
    
    def _create_qa_pairs(self) -> list[dict[str, str]]:
        """Generate predefined Q&A pairs from dataset"""
        df = self.df
        
        # Calculate statistics
        total_production = int(df['Production'].sum())
        total_area = int(df['Area'].sum())
        years = sorted(df['Crop_Year'].unique())
        states = sorted(df['State_Name'].unique())
        crops = sorted(df['Crop'].unique())
        
        # Top statistics
        top_states = df.groupby('State_Name')['Production'].sum().nlargest(3)
        top_crops = df.groupby('Crop')['Production'].sum().nlargest(3)
        top_yield = df.groupby('Crop')['Production'].mean().nlargest(3)
        
        qa_pairs = [
            # Basic Info
            {
                'question': 'What is this application',
                'answer': 'AgriVision AI is an intelligent agricultural assistant that provides insights about crop production, yields, and farming data across Indian states. Ask me about crops, production statistics, or seasonal trends!'
            },
            {
                'question': 'Hello',
                'answer': 'Hello! I\'m AgriVision AI, your agricultural assistant. I can help you with crop data, production statistics, and farming insights. What would you like to know?'
            },
            {
                'question': 'Hi there',
                'answer': 'Hello! Welcome to AgriVision AI. I can answer questions about crop production, yields, regional farming data, and seasonal trends. What\'s your question?'
            },
            
            # Production Questions
            {
                'question': 'What is the total agricultural production',
                'answer': f'The total agricultural production in our dataset is {total_production:,} units across all states and years ({years[0]}-{years[-1]}).'
            },
            {
                'question': 'Which state produces the most',
                'answer': f'Top 3 producing states:\n1. {top_states.index[0]}: {int(top_states.iloc[0]):,} units\n2. {top_states.index[1]}: {int(top_states.iloc[1]):,} units\n3. {top_states.index[2]}: {int(top_states.iloc[2]):,} units'
            },
            {
                'question': 'What are the most productive crops',
                'answer': f'Top 3 most productive crops:\n1. {top_crops.index[0]}: {int(top_crops.iloc[0]):,} units\n2. {top_crops.index[1]}: {int(top_crops.iloc[1]):,} units\n3. {top_crops.index[2]}: {int(top_crops.iloc[2]):,} units'
            },
            {
                'question': 'What is the highest yield crop',
                'answer': f'Crops with highest average yield:\n1. {top_yield.index[0]}: {top_yield.iloc[0]:.2f} units/area\n2. {top_yield.index[1]}: {top_yield.iloc[1]:.2f} units/area\n3. {top_yield.index[2]}: {top_yield.iloc[2]:.2f} units/area'
            },
            
            # Year-based Questions
            {
                'question': 'What crops were grown in 2011 in west bengal',
                'answer': self._get_crops_by_year_state(2011, 'West Bengal')
            },
            {
                'question': 'What crops were grown in 2010',
                'answer': self._get_crops_by_year(2010)
            },
            {
                'question': 'What was the production in 2015',
                'answer': self._get_production_by_year(2015)
            },
            {
                'question': 'Which year had maximum production',
                'answer': self._get_max_production_year()
            },
            
            # State-based Questions
            {
                'question': 'What is produced in west bengal',
                'answer': self._get_crops_by_state('West Bengal')
            },
            {
                'question': 'What is produced in maharashtra',
                'answer': self._get_crops_by_state('Maharashtra')
            },
            {
                'question': 'What is produced in tamil nadu',
                'answer': self._get_crops_by_state('Tamil Nadu')
            },
            {
                'question': 'What is produced in punjab',
                'answer': self._get_crops_by_state('Punjab')
            },
            
            # Crop-based Questions
            {
                'question': 'Tell me about rice production',
                'answer': self._get_crop_info('Rice')
            },
            {
                'question': 'Tell me about wheat',
                'answer': self._get_crop_info('Wheat')
            },
            {
                'question': 'What about sugarcane',
                'answer': self._get_crop_info('Sugarcane')
            },
            {
                'question': 'Tell me about cotton',
                'answer': self._get_crop_info('Cotton')
            },
            
            # Season-based Questions
            {
                'question': 'What are seasonal trends',
                'answer': self._get_seasonal_trends()
            },
            {
                'question': 'What is kharif season',
                'answer': 'Kharif season is the monsoon season (June-October) in India. Crops like rice, cotton, and sugarcane are primarily grown during this season.'
            },
            {
                'question': 'What is rabi season',
                'answer': 'Rabi season is the winter season (October-March) in India. Crops like wheat, barley, and pulses are primarily grown during this season.'
            },
            
            # Data Overview
            {
                'question': 'How many states are in the data',
                'answer': f'Our dataset includes data from {len(states)} states across India.'
            },
            {
                'question': 'How many crops are in the database',
                'answer': f'The database contains information about {len(crops)} different crops.'
            },
            {
                'question': 'What years does the data cover',
                'answer': f'The dataset covers crop data from {years[0]} to {years[-1]}.'
            },
            {
                'question': 'Tell me about the dataset',
                'answer': f'AgriVision dataset contains:\n- {len(states)} States\n- {len(crops)} Crops\n- Years: {years[0]}-{years[-1]}\n- Total Production: {total_production:,} units\n- Total Area: {total_area:,} units'
            },
            
            # Help
            {
                'question': 'What can I ask you',
                'answer': 'You can ask me about:\n- Crop production statistics\n- Which crops are grown in specific states\n- Top producing states and crops\n- Seasonal farming information\n- Crop yields and production trends\n- Regional agricultural data\n\nTry asking: "Which state produces the most?" or "What crops were grown in 2011 in West Bengal?"'
            },
            {
                'question': 'Help',
                'answer': 'I can help you with agricultural data! Ask me about:\n- Crop production by state\n- Top crops and regions\n- Seasonal trends\n- Crop statistics and yields\n\nExample: "What is produced in Maharashtra?" or "Tell me about rice production"'
            },
        ]
        
        return qa_pairs
    
    def _get_crops_by_year_state(self, year: int, state: str) -> str:
        """Get crops grown in a specific year and state"""
        filtered = self.df[(self.df['Crop_Year'] == year) & (self.df['State_Name'].str.contains(state, case=False, na=False))]
        if filtered.empty:
            return f"No data found for {year} in {state}."
        
        crops = filtered['Crop'].unique()
        production = filtered['Production'].sum()
        return f"In {year}, {state} grew: {', '.join(crops[:5])}{'...' if len(crops) > 5 else ''}. Total production: {int(production):,} units."
    
    def _get_crops_by_year(self, year: int) -> str:
        """Get major crops grown in a year"""
        filtered = self.df[self.df['Crop_Year'] == year]
        if filtered.empty:
            return f"No data found for year {year}."
        
        top_crops = filtered.groupby('Crop')['Production'].sum().nlargest(5)
        result = f"Major crops grown in {year}:\n"
        for i, (crop, prod) in enumerate(top_crops.items(), 1):
            result += f"{i}. {crop}: {int(prod):,} units\n"
        return result.strip()
    
    def _get_production_by_year(self, year: int) -> str:
        """Get total production in a year"""
        filtered = self.df[self.df['Crop_Year'] == year]
        if filtered.empty:
            return f"No data found for year {year}."
        
        total = int(filtered['Production'].sum())
        return f"Total agricultural production in {year} was {total:,} units."
    
    def _get_max_production_year(self):
        """Get year with maximum production"""
        yearly = self.df.groupby('Crop_Year')['Production'].sum()
        max_year = yearly.idxmax()
        max_prod = int(yearly.max())
        return f"The year {max_year} had maximum production with {max_prod:,} units."
    
    def _get_crops_by_state(self, state: str) -> str:
        """Get crops grown in a state"""
        filtered = self.df[self.df['State_Name'].str.contains(state, case=False, na=False)]
        if filtered.empty:
            return f"No data found for {state}."
        
        top_crops = filtered.groupby('Crop')['Production'].sum().nlargest(5)
        result = f"Top crops in {state}:\n"
        for i, (crop, prod) in enumerate(top_crops.items(), 1):
            result += f"{i}. {crop}: {int(prod):,} units\n"
        return result.strip()
    
    def _get_crop_info(self, crop: str) -> str:
        """Get information about a crop"""
        filtered = self.df[self.df['Crop'].str.contains(crop, case=False, na=False)]
        if filtered.empty:
            return f"No data found for {crop}."
        
        total_prod = int(filtered['Production'].sum())
        states = filtered['State_Name'].nunique()
        avg_yield = filtered['Production'].mean()
        return f"{crop} statistics:\n- Total Production: {total_prod:,} units\n- Grown in {states} states\n- Average Yield: {avg_yield:.2f} units\n- Years: {filtered['Crop_Year'].min():.0f}-{filtered['Crop_Year'].max():.0f}"
    
    def _get_seasonal_trends(self) -> str:
        """Get seasonal production trends"""
        seasonal = self.df.groupby('Season')['Production'].sum()
        result = "Seasonal production trends:\n"
        for season, prod in seasonal.items():
            result += f"- {season.strip()}: {int(prod):,} units\n"
        return result.strip()
    
    def get_response(self, user_query: str) -> str:
        """
        Get the best matching answer for a user query using TF-IDF cosine similarity
        """
        if not user_query.strip():
            return "Please ask me something about agriculture!"
        
        # Vectorize user query
        user_vector = self.vectorizer.transform([user_query])
        
        # Calculate similarity with all questions
        similarities = cosine_similarity(user_vector, self.question_vectors)[0]
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # If similarity is too low, provide a default response
        if best_score < 0.2:
            return "I'm not sure about that. Try asking about crop production, states, yields, or seasonal trends. You can also ask 'What can I ask you?' for more options."
        
        return self.answers[best_idx]
