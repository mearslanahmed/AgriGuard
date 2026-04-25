import React from 'react';
import {
  View, Text, StyleSheet, Image,
  ScrollView, TouchableOpacity
} from 'react-native';

const getConfidenceColor = (confidence) => {
  if (confidence >= 85) return '#2e7d32';
  if (confidence >= 60) return '#f57c00';
  return '#e53935';
};

export default function ResultScreen({ navigation, route }) {
  const { mlResult, pesticideData, imageUri } = route.params;
  const { confidence, disease, crop, is_healthy } = mlResult;

  // Pesticide info — only present if disease detected and data found in DB
  const p = pesticideData?.pesticide;

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
        <Text style={styles.backText}>← Back</Text>
      </TouchableOpacity>

      <Text style={styles.title}>Detection Result</Text>

      {imageUri && <Image source={{ uri: imageUri }} style={styles.image} />}

      {/* Main result card */}
      <View style={styles.card}>
        <Text style={styles.label}>Crop</Text>
        <Text style={styles.disease}>{crop}</Text>

        <Text style={styles.label}>Detected Condition</Text>
        <Text style={styles.disease}>{disease}</Text>

        <Text style={styles.label}>Confidence</Text>
        <Text style={[styles.confidence, { color: getConfidenceColor(confidence) }]}>
          {confidence.toFixed(1)}%
        </Text>

        <View style={[styles.badge, { backgroundColor: is_healthy ? '#e8f5e9' : '#fdecea' }]}>
          <Text style={[styles.badgeText, { color: is_healthy ? '#2e7d32' : '#e53935' }]}>
            {is_healthy ? 'Plant is Healthy' : 'Disease Detected'}
          </Text>
        </View>
      </View>

      {/* Pesticide advisory — only shown when disease detected */}
      {!is_healthy && p && (
        <View style={styles.pesticideCard}>
          <Text style={styles.pesticideTitle}>Pesticide Advisory</Text>

          <View style={styles.row}>
            <Text style={styles.fieldLabel}>Recommended Pesticide</Text>
            <Text style={styles.fieldValue}>{p.name}</Text>
          </View>

          <View style={styles.divider} />

          <View style={styles.row}>
            <Text style={styles.fieldLabel}>Dosage</Text>
            <Text style={styles.fieldValue}>{p.dosage}</Text>
          </View>

          <View style={styles.divider} />

          <View style={styles.row}>
            <Text style={styles.fieldLabel}>Spray Interval</Text>
            <Text style={styles.fieldValue}>{p.spray_interval}</Text>
          </View>

          <View style={styles.divider} />

          <View style={styles.row}>
            <Text style={styles.fieldLabel}>Water Ratio</Text>
            <Text style={styles.fieldValue}>{p.water_ratio}</Text>
          </View>

          <View style={styles.divider} />

          {/* Safety warning — highlighted separately */}
          <View style={styles.safetyBox}>
            <Text style={styles.safetyLabel}>Safety Instructions</Text>
            <Text style={styles.safetyText}>{p.safety}</Text>
          </View>

          {p.notes && (
            <View style={styles.notesBox}>
              <Text style={styles.notesLabel}>Farmer Notes</Text>
              <Text style={styles.notesText}>{p.notes}</Text>
            </View>
          )}
        </View>
      )}

      {/* Healthy message */}
      {is_healthy && (
        <View style={styles.healthyCard}>
          <Text style={styles.healthyText}>
            Your plant looks healthy! No pesticide treatment needed.
            Keep monitoring regularly.
          </Text>
        </View>
      )}

      {/* No pesticide data fallback */}
      {!is_healthy && !p && (
        <View style={styles.infoCard}>
          <Text style={styles.infoTitle}>Advisory Unavailable</Text>
          <Text style={styles.infoText}>
            No pesticide data found for this disease. Please consult a local agricultural expert.
          </Text>
        </View>
      )}

      <TouchableOpacity style={styles.button} onPress={() => navigation.navigate('Detect')}>
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
  backText: { color: '#2e7d32', fontSize: 15 },
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
    fontSize: 20,
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
  badgeText: { fontWeight: '600', fontSize: 14 },
  pesticideCard: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 20,
    width: '100%',
    marginBottom: 16,
    elevation: 2,
  },
  pesticideTitle: {
    fontSize: 17,
    fontWeight: 'bold',
    color: '#e65100',
    marginBottom: 12,
  },
  row: { marginBottom: 8 },
  fieldLabel: { fontSize: 12, color: '#888', marginBottom: 2 },
  fieldValue: { fontSize: 15, color: '#333', fontWeight: '500' },
  divider: { height: 1, backgroundColor: '#f0f0f0', marginVertical: 8 },
  safetyBox: {
    backgroundColor: '#fdecea',
    borderRadius: 10,
    padding: 12,
    marginTop: 8,
  },
  safetyLabel: {
    fontWeight: 'bold',
    color: '#e53935',
    marginBottom: 4,
    fontSize: 13,
  },
  safetyText: { color: '#555', fontSize: 13, lineHeight: 20 },
  notesBox: {
    backgroundColor: '#e8f5e9',
    borderRadius: 10,
    padding: 12,
    marginTop: 8,
  },
  notesLabel: {
    fontWeight: 'bold',
    color: '#2e7d32',
    marginBottom: 4,
    fontSize: 13,
  },
  notesText: { color: '#555', fontSize: 13, lineHeight: 20 },
  healthyCard: {
    backgroundColor: '#e8f5e9',
    borderRadius: 14,
    padding: 16,
    width: '100%',
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#2e7d32',
  },
  healthyText: { color: '#2e7d32', fontSize: 14, lineHeight: 22 },
  infoCard: {
    backgroundColor: '#fff3e0',
    borderRadius: 14,
    padding: 16,
    width: '100%',
    marginBottom: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#f57c00',
  },
  infoTitle: { fontWeight: 'bold', color: '#e65100', marginBottom: 6, fontSize: 15 },
  infoText: { color: '#555', fontSize: 13, lineHeight: 20 },
  button: {
    backgroundColor: '#2e7d32',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
    width: '100%',
    marginTop: 8,
  },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});