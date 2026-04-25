import React from 'react';
import {
  View, Text, StyleSheet, Image,
  ScrollView, TouchableOpacity
} from 'react-native';

// Cleaning class names:  "Tomato___Early_blight" to "Tomato Early Blight"
const formatClassName = (name) => {
  return name.replace(/_{2,}/g, ' ').replace(/_/g, ' ')
    .split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
};

const getConfidenceColor = (confidence) => {
    if (confidence >= 0.85) return '#2e7d32'; // green — high confidence
    if (confidence >= 0.60) return '#f57c00'; // orange — medium
    return '#e53935';                          // red — low
};

export default function ResultScreen({ navigation, route }) {
    const { result, imageUri } = route.params;
    const { class_name, confidence, disease, crop, is_healthy } = result;
    const formattedName = formatClassName(class_name);
    const confidencePercent = (confidence).toFixed(1);
    const isHealthy = is_healthy;


  return (
    <ScrollView contentContainerStyle={styles.container}>
        {/* Back button */}
        <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
            <Text style={styles.backText}>← Back</Text>
        </TouchableOpacity>

        <Text style={styles.title}>Detection Result</Text>

        {/* Image */}
        {imageUri && (
            <Image source = {{uri: imageUri}} style={styles.image} />
        )}

        {/* Result card */}
        <View style={styles.card}>
            <Text style={styles.label}>Disease Detected</Text>
            <Text style={styles.disease}>{crop} — {disease}</Text>

            <Text style={styles.label}>Confidence</Text>
            <Text style={[styles.confidence, {color: getConfidenceColor(confidence)}]}>
                {confidencePercent}%
            </Text>

            {/* status badge */}
             <View style={[styles.badge, { backgroundColor: isHealthy ? '#e8f5e9' : '#fdecea' }]}>
          <Text style={[styles.badgeText, { color: isHealthy ? '#2e7d32' : '#e53935' }]}>
            {isHealthy ? 'Plant is Healthy' : 'Disease Detected'}
          </Text>
        </View>
      </View>

      {/* TODO: Placeholder for pesticide info - after backend setup*/}
      {!isHealthy && (
        <View style={styles.infoCard}>
          <Text style={styles.infoTitle}>Pesticide Advisory</Text>
          <Text style={styles.infoText}>
            Pesticide recommendations will be shown here once connected to the backend database.
          </Text>
        </View>
      )}

      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Detect')}
        >
        <Text style={styles.buttonText}>Scan Another</Text>
        </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#f5f5f5',
    padding: 24,
    alignItems: 'center',
  },
  backButton: {
    alignSelf: 'flex-start',
    marginBottom: 10,
    marginTop: 10,
  },
  backText: {
    color: '#2e7d32',
    fontSize: 15,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2e7d32',
    marginBottom: 16,
  },
  image: {
    width: '100%',
    height: 220,
    borderRadius: 14,
    resizeMode: 'cover',
    marginBottom: 20,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 20,
    width: '100%',
    marginBottom: 16,
    alignItems: 'center',
    elevation: 2,
  },
  label: {
    fontSize: 12,
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginTop: 12,
  },
  disease: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginTop: 4,
  },
  confidence: {
    fontSize: 36,
    fontWeight: 'bold',
    marginTop: 4,
  },
  badge: {
    marginTop: 16,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  badgeText: {
    fontWeight: '600',
    fontSize: 14,
  },
  infoCard: {
    backgroundColor: '#fff3e0',
    borderRadius: 14,
    padding: 16,
    width: '100%',
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#f57c00',
  },
  infoTitle: {
    fontWeight: 'bold',
    color: '#e65100',
    marginBottom: 6,
    fontSize: 15,
  },
  infoText: {
    color: '#555',
    fontSize: 13,
    lineHeight: 20,
  },
  button: {
    backgroundColor: '#2e7d32',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
    width: '100%',
    marginTop: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});