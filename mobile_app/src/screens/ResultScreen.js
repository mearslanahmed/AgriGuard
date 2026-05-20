import React, { useEffect, useRef } from 'react';
import {
  View, Text, StyleSheet, Image,
  ScrollView, TouchableOpacity, Animated, Easing, StatusBar
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

const getConfidenceColor = (confidence) => {
  if (confidence >= 85) return '#2e7d32';
  if (confidence >= 60) return '#f57c00';
  return '#e53935';
};

const getConfidenceBg = (confidence) => {
  if (confidence >= 85) return '#e8f5e9';
  if (confidence >= 60) return '#fff3e0';
  return '#fdecea';
};

export default function ResultScreen({ navigation, route }) {
  const { mlResult, pesticideData, imageUri } = route.params;
  const { confidence, disease, crop, is_healthy } = mlResult;
  const p = pesticideData?.pesticide;

  // Animations
  const headerAnim = useRef(new Animated.Value(0)).current;
  const imageAnim = useRef(new Animated.Value(0)).current;
  const card1Anim = useRef(new Animated.Value(0)).current;
  const card2Anim = useRef(new Animated.Value(0)).current;
  const btnAnim = useRef(new Animated.Value(0)).current;
  const confidenceAnim = useRef(new Animated.Value(0)).current;
  const scanBtnScale = useRef(new Animated.Value(1)).current;
  const backBtnScale = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Animated.stagger(80, [
      Animated.timing(headerAnim, {
        toValue: 1, duration: 350,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(imageAnim, {
        toValue: 1, duration: 400,
        easing: Easing.out(Easing.back(1.1)),
        useNativeDriver: true,
      }),
      Animated.timing(card1Anim, {
        toValue: 1, duration: 380,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(card2Anim, {
        toValue: 1, duration: 380,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(btnAnim, {
        toValue: 1, duration: 350,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
    ]).start();

    // Confidence counter animation
    Animated.timing(confidenceAnim, {
      toValue: confidence,
      duration: 900,
      delay: 400,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false,
    }).start();
  }, []);

  const pressIn = (scale) =>
    Animated.spring(scale, { toValue: 0.94, useNativeDriver: true }).start();
  const pressOut = (scale) =>
    Animated.spring(scale, { toValue: 1, friction: 4, useNativeDriver: true }).start();

  const handleScanAnother = () => {
    // Navigate back and reset image state
    navigation.goBack();
  };

  const animStyle = (anim, slideFrom = 20) => ({
    opacity: anim,
    transform: [{
      translateY: anim.interpolate({
        inputRange: [0, 1], outputRange: [slideFrom, 0]
      })
    }]
  });

  return (
    <View style={styles.screen}>
      <StatusBar barStyle="dark-content" backgroundColor="#f7faf7" />

      {/* Fixed header */}
      <Animated.View style={[styles.header, animStyle(headerAnim, -10)]}>
        <Animated.View style={{ transform: [{ scale: backBtnScale }] }}>
          <TouchableOpacity
            style={styles.backBtn}
            onPress={() => navigation.goBack()}
            onPressIn={() => pressIn(backBtnScale)}
            onPressOut={() => pressOut(backBtnScale)}
            activeOpacity={1}
          >
            <Ionicons name="chevron-back" size={20} color="#2e7d32" />
          </TouchableOpacity>
        </Animated.View>
        <Text style={styles.headerTitle}>Result</Text>
        <View style={styles.headerSpacer} />
      </Animated.View>

      <ScrollView
        contentContainerStyle={styles.container}
        showsVerticalScrollIndicator={false}
      >
        {/* Crop image */}
        {imageUri && (
          <Animated.View style={[styles.imageWrapper, animStyle(imageAnim)]}>
            <Image source={{ uri: imageUri }} style={styles.image} />
            {/* Status pill over image */}
            <View style={[
              styles.statusPill,
              { backgroundColor: is_healthy ? '#2e7d32' : '#e53935' }
            ]}>
              <Ionicons
                name={is_healthy ? 'checkmark-circle' : 'warning'}
                size={13}
                color="#fff"
                style={{ marginRight: 5 }}
              />
              <Text style={styles.statusPillText}>
                {is_healthy ? 'Healthy' : 'Disease Detected'}
              </Text>
            </View>
          </Animated.View>
        )}

        {/* Main result card */}
        <Animated.View style={[styles.card, animStyle(card1Anim)]}>
          {/* Crop + disease row */}
          <View style={styles.infoRow}>
            <View style={styles.infoCell}>
              <Text style={styles.cellLabel}>Crop</Text>
              <Text style={styles.cellValue}>{crop}</Text>
            </View>
            <View style={styles.cellDivider} />
            <View style={styles.infoCell}>
              <Text style={styles.cellLabel}>Condition</Text>
              <Text style={styles.cellValue} numberOfLines={2}>{disease}</Text>
            </View>
          </View>

          <View style={styles.cardDivider} />

          {/* Confidence bar */}
          <View style={styles.confidenceSection}>
            <View style={styles.confidenceHeader}>
              <Text style={styles.cellLabel}>Confidence</Text>
              <Animated.Text style={[
                styles.confidenceNumber,
                { color: getConfidenceColor(confidence) }
              ]}>
                {confidenceAnim.interpolate({
                  inputRange: [0, confidence],
                  outputRange: ['0.0%', `${confidence.toFixed(1)}%`]
                })}
              </Animated.Text>
            </View>
            <View style={styles.barTrack}>
              <Animated.View style={[
                styles.barFill,
                {
                  backgroundColor: getConfidenceColor(confidence),
                  width: confidenceAnim.interpolate({
                    inputRange: [0, 100],
                    outputRange: ['0%', '100%']
                  })
                }
              ]} />
            </View>
          </View>
        </Animated.View>

        {/* Pesticide advisory */}
        {!is_healthy && p && (
          <Animated.View style={[styles.card, animStyle(card2Anim)]}>
            <View style={styles.advisoryHeader}>
              <View style={styles.advisoryIconBox}>
                <Ionicons name="flask-outline" size={16} color="#e65100" />
              </View>
              <Text style={styles.advisoryTitle}>Pesticide Advisory</Text>
            </View>

            <View style={styles.cardDivider} />

            {[
              { label: 'Recommended Pesticide', value: p.name },
              { label: 'Dosage', value: p.dosage },
              { label: 'Spray Interval', value: p.spray_interval },
              { label: 'Water Ratio', value: p.water_ratio },
            ].map((item, i) => (
              <View key={i}>
                <View style={styles.advisoryRow}>
                  <Text style={styles.cellLabel}>{item.label}</Text>
                  <Text style={styles.advisoryValue}>{item.value}</Text>
                </View>
                {i < 3 && <View style={styles.rowDivider} />}
              </View>
            ))}

            {/* Safety box */}
            <View style={styles.safetyBox}>
              <View style={styles.safetyHeader}>
                <Ionicons name="shield-outline" size={14} color="#e53935" />
                <Text style={styles.safetyLabel}>Safety Instructions</Text>
              </View>
              <Text style={styles.safetyText}>{p.safety}</Text>
            </View>

            {p.notes && (
              <View style={styles.notesBox}>
                <View style={styles.safetyHeader}>
                  <Ionicons name="document-text-outline" size={14} color="#2e7d32" />
                  <Text style={styles.notesLabel}>Farmer Notes</Text>
                </View>
                <Text style={styles.notesText}>{p.notes}</Text>
              </View>
            )}
          </Animated.View>
        )}

        {/* Healthy message */}
        {is_healthy && (
          <Animated.View style={[styles.healthyCard, animStyle(card2Anim)]}>
            <Ionicons name="leaf" size={22} color="#2e7d32" style={{ marginBottom: 8 }} />
            <Text style={styles.healthyTitle}>Plant looks great!</Text>
            <Text style={styles.healthyText}>
              No pesticide treatment needed. Keep monitoring regularly.
            </Text>
          </Animated.View>
        )}

        {/* No data fallback */}
        {!is_healthy && !p && (
          <Animated.View style={[styles.infoCard, animStyle(card2Anim)]}>
            <Ionicons name="information-circle-outline" size={20} color="#f57c00" style={{ marginBottom: 6 }} />
            <Text style={styles.infoTitle}>Advisory Unavailable</Text>
            <Text style={styles.infoText}>
              No pesticide data found for this disease. Consult a local agricultural expert.
            </Text>
          </Animated.View>
        )}

        {/* Scan another */}
        <Animated.View style={[animStyle(btnAnim), { marginTop: 8, marginBottom: 16 }]}>
          <Animated.View style={{ transform: [{ scale: scanBtnScale }] }}>
            <TouchableOpacity
              style={styles.scanBtn}
              onPress={handleScanAnother}
              onPressIn={() => pressIn(scanBtnScale)}
              onPressOut={() => pressOut(scanBtnScale)}
              activeOpacity={1}
            >
              <Ionicons name="scan-outline" size={18} color="#fff" style={{ marginRight: 8 }} />
              <Text style={styles.scanBtnText}>Scan Another</Text>
            </TouchableOpacity>
          </Animated.View>
        </Animated.View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#f7faf7',
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingTop: 32,
    paddingBottom: 10,
    backgroundColor: '#f7faf7',
  },
  backBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#e8f5e9',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    flex: 1,
    textAlign: 'center',
    fontSize: 17,
    fontWeight: '600',
    color: '#1b5e20',
  },
  headerSpacer: { width: 36 },

  container: {
    paddingHorizontal: 16,
    paddingBottom: 110,
  },

  // Image
  imageWrapper: {
    position: 'relative',
    marginBottom: 14,
  },
  image: {
    width: '100%',
    height: 210,
    borderRadius: 16,
    resizeMode: 'cover',
  },
  statusPill: {
    position: 'absolute',
    bottom: 12,
    left: 12,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  statusPillText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },

  // Cards
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 18,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: '#e8f0e8',
  },
  cardDivider: {
    height: 1,
    backgroundColor: '#f0f4f0',
    marginVertical: 14,
  },
  rowDivider: {
    height: 1,
    backgroundColor: '#f5f5f5',
    marginVertical: 2,
  },

  // Info row (crop + condition)
  infoRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  infoCell: {
    flex: 1,
  },
  cellDivider: {
    width: 1,
    backgroundColor: '#f0f0f0',
    marginHorizontal: 16,
    alignSelf: 'stretch',
  },
  cellLabel: {
    fontSize: 11,
    color: '#999',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 4,
  },
  cellValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#222',
    lineHeight: 22,
  },

  // Confidence
  confidenceSection: {},
  confidenceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  confidenceNumber: {
    fontSize: 22,
    fontWeight: '700',
  },
  barTrack: {
    height: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 4,
  },

  // Advisory
  advisoryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  advisoryIconBox: {
    width: 30,
    height: 30,
    borderRadius: 8,
    backgroundColor: '#fff3e0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  advisoryTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#e65100',
  },
  advisoryRow: {
    paddingVertical: 10,
  },
  advisoryValue: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
    marginTop: 3,
  },
  safetyBox: {
    backgroundColor: '#fdecea',
    borderRadius: 10,
    padding: 12,
    marginTop: 12,
  },
  safetyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 6,
  },
  safetyLabel: {
    fontWeight: '700',
    color: '#e53935',
    fontSize: 13,
  },
  safetyText: {
    color: '#555',
    fontSize: 13,
    lineHeight: 20,
  },
  notesBox: {
    backgroundColor: '#e8f5e9',
    borderRadius: 10,
    padding: 12,
    marginTop: 8,
  },
  notesLabel: {
    fontWeight: '700',
    color: '#2e7d32',
    fontSize: 13,
  },
  notesText: {
    color: '#555',
    fontSize: 13,
    lineHeight: 20,
  },

  // Healthy
  healthyCard: {
    backgroundColor: '#e8f5e9',
    borderRadius: 16,
    padding: 20,
    marginBottom: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#c8e6c9',
  },
  healthyTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#2e7d32',
    marginBottom: 6,
  },
  healthyText: {
    color: '#4caf50',
    fontSize: 13,
    lineHeight: 20,
    textAlign: 'center',
  },

  // No data
  infoCard: {
    backgroundColor: '#fff8f0',
    borderRadius: 16,
    padding: 20,
    marginBottom: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ffe0b2',
  },
  infoTitle: {
    fontWeight: '700',
    color: '#e65100',
    fontSize: 15,
    marginBottom: 6,
  },
  infoText: {
    color: '#555',
    fontSize: 13,
    lineHeight: 20,
    textAlign: 'center',
  },

  // Scan another
  scanBtn: {
    backgroundColor: '#2e7d32',
    paddingVertical: 15,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  scanBtnText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});