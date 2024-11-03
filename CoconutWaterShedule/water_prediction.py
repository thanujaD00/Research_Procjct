import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import os

def create_output_directories():
    """Create directories for outputs if they don't exist"""
    # Create main output directory
    output_dir = 'model_outputs'
    images_dir = os.path.join(output_dir, 'visualizations')
    models_dir = os.path.join(output_dir, 'models')
    
    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Created output directories:")
    print(f"- Images: {images_dir}")
    print(f"- Models: {models_dir}")
    
    return output_dir, images_dir, models_dir

# Load and prepare data
def prepare_data(csv_file, images_dir):
    try:
        # Load dataset
        df = pd.read_csv(csv_file)
        print("Data loaded successfully")
        
        # Plot data distribution before categorization
        plt.figure(figsize=(10, 6))
        plt.hist(df['Water Level (liters)'], bins=30)
        plt.title('Distribution of Water Levels')
        plt.xlabel('Water Level (liters)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(images_dir, 'water_level_distribution.png'))
        plt.close()
        
        # Convert water levels to categories
        def categorize_water_level(water_level):
            if 50 <= water_level <= 100:
                return 1    # High water need
            elif 30 <= water_level < 50:
                return 2    # Moderate water need
            elif 10 <= water_level < 30:
                return 3    # Low water need
            else:
                return 0    # No water needed
        
        df['Water Level Category'] = df['Water Level (liters)'].apply(categorize_water_level)
        
        # Plot category distribution
        plt.figure(figsize=(10, 6))
        df['Water Level Category'].value_counts().plot(kind='bar')
        plt.title('Distribution of Water Level Categories')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(range(4), ['No Need', 'High', 'Moderate', 'Low'])
        plt.savefig(os.path.join(images_dir, 'category_distribution.png'))
        plt.close()
        
        # Prepare features
        X = df.drop(['Date','Rainfall (mm)','Irrigation Status (0/1)',
                    'Water Level (liters)','Water Level Category'], axis=1)
        y = df['Water Level Category']
        
        # Correlation matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'correlation_matrix.png'))
        plt.close()
        
        # Handle imbalanced data
        smote = SMOTE(sampling_strategy='auto')
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Plot balanced category distribution
        plt.figure(figsize=(10, 6))
        pd.Series(y_resampled).value_counts().plot(kind='bar')
        plt.title('Distribution of Water Level Categories (After SMOTE)')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(range(4), ['No Need', 'High', 'Moderate', 'Low'])
        plt.savefig(os.path.join(images_dir, 'balanced_category_distribution.png'))
        plt.close()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error in data preparation: {str(e)}")
        return None, None, None, None

# Train and evaluate models
def train_models(X_train, X_test, y_train, y_test, images_dir, models_dir):
    try:
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # Store accuracies for plotting
        accuracies = []
        model_names = []
        
        # Create subplot figure
        fig = plt.figure(figsize=(15, 10))
        
        for i, (name, model) in enumerate(models.items()):
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            model_names.append(name)
            
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Plot confusion matrix
            plt.subplot(2, 2, i+1)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No Need', 'High', 'Moderate', 'Low'],
                       yticklabels=['No Need', 'High', 'Moderate', 'Low'])
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            # Save model
            model_filename = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
            joblib.dump(model, model_filename)
            print(f"Model saved as: {model_filename}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'confusion_matrices.png'))
        plt.close()
        
        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies)
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'accuracy_comparison.png'))
        plt.close()
        
        # Plot feature importance for Random Forest
        rf_model = models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'feature_importance.png'))
        plt.close()
        
        return models
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        return None

# Make predictions
def predict_water_need(model_file, new_data):
    try:
        # Load model
        model = joblib.load(model_file)
        
        # Make prediction
        prediction = model.predict(new_data)[0]
        
        # Convert prediction to readable format
        categories = {
            0: "0 - 0 (No water needed)",
            1: "50 - 100 (High water need)",
            2: "30 - 50 (Moderate water need)",
            3: "10 - 30 (Low water need)"
        }
        
        return categories.get(prediction, "Invalid prediction")
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

def visualize_predictions(new_data, prediction, images_dir):
    try:
        # Create a radar chart for input features
        features = new_data.columns
        values = new_data.iloc[0].values
        
        # Normalize values for radar chart
        normalized_values = (values - values.min()) / (values.max() - values.min())
        
        angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
        
        # Close the plot by appending first value
        values_plot = np.concatenate((normalized_values, [normalized_values[0]]))
        angles_plot = np.concatenate((angles, [angles[0]]))
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles_plot, values_plot)
        ax.fill(angles_plot, values_plot, alpha=0.25)
        plt.xticks(angles, features, size=8)
        plt.title('Input Features Radar Chart')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'input_features_radar.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error in prediction visualization: {str(e)}")

if __name__ == "__main__":
    # Create output directories
    output_dir, images_dir, models_dir = create_output_directories()
    
    # File path
    csv_file = 'coconut_irrigation_data_with_water_level_8.csv'
    
    # Prepare data
    print("Starting data preparation...")
    X_train, X_test, y_train, y_test = prepare_data(csv_file, images_dir)
    
    if X_train is not None:
        # Train models and create visualizations
        print("\nStarting model training...")
        models = train_models(X_train, X_test, y_train, y_test, images_dir, models_dir)
        
        if models:
            # Example prediction
            new_data = pd.DataFrame({
                'Soil Moisture (10 cm) (%)': [14.477],
                'Soil Moisture (20 cm) (%)': [24.64],
                'Soil Moisture (30 cm) (%)': [36.25],
                'Plant Age (years)': [5],
                'Temperature (°C)': [28.3],
                'Humidity (%)': [72.52],
                'Rain Status (0/1)': [0]
            })
            
            print("\nMaking prediction with Random Forest model...")
            model_path = os.path.join(models_dir, 'random_forest_model.pkl')
            prediction = predict_water_need(model_path, new_data)
            print(f"Predicted water need: {prediction}")
            
            # Visualize the prediction input
            visualize_predictions(new_data, prediction, images_dir)
            
            print("\nAll visualizations have been saved in:", images_dir)
            print("All models have been saved in:", models_dir)