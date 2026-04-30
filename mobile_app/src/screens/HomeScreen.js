import React, { useState, useCallback, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  ScrollView, ActivityIndicator, RefreshControl,
  Animated
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import * as SecureStore from 'expo-secure-store';
import { BACKEND_URL } from '../config';

export default function HomeScreen({ navigation }) {
  const { userInfo } = useAuth();
  const [recentScan, setRecentScan] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Animations
  const heroAnim = useRef(new Animated.Value(0)).current;
  const cardAnim = useRef(new Animated.Value(0)).current;
  const scanAnim = useRef(new Animated.Value(1)).current;

  useFocusEffect(
    useCallback(() => {
      fetchRecentScan();
      runEntryAnimation();
    }, [])
  );

  const runEntryAnimation = () => {
    // Reset first
    heroAnim.setValue(0);
    cardAnim.setValue(0);

    Animated.stagger(200, [
      Animated.timing(heroAnim, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
      Animated.timing(cardAnim, {
        toValue: 1,
        duration: 350,
        useNativeDriver: true,
      }),
    ]).start();
  };

  // Pulse animation on scan button press
  const pulseScanButton = () => {
    Animated.sequence([
      Animated.timing(scanAnim, {
        toValue: 0.95,
        duration: 150,
        useNativeDriver: true,
      }),
      Animated.spring(scanAnim, {
        toValue: 1,
        tension: 80,
        friction: 6,
        useNativeDriver: true,
      }),
    ]).start();
    navigation.navigate('Detect');
  };

  const fetchRecentScan = async () => {
    try {
      const token = await SecureStore.getItemAsync('userToken');
      const response = await fetch(`${BACKEND_URL}/api/scans?limit=1`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        const data = await response.json();
        setRecentScan(data[0] || null);
      }
    } catch (err) {
      console.log('Home fetch error:', err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const firstName = userInfo?.name?.split(' ')[0] || 'Farmer';

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  };

  const getTip = () => {
    const tips = [
      'Early detection saves up to 40% of crop yield.',
      'Scan leaves in natural daylight for best accuracy.',
      'Wet leaves can affect detection — scan when dry.',
      'Check your crops at least once a week.',
      'Yellowing edges often signal nutrient deficiency.',
    ];
    return tips[new Date().getDay() % tips.length];
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={() => { setRefreshing(true); fetchRecentScan(); }}
          colors={['#2e7d32']}
        />
      }
      showsVerticalScrollIndicator={false}
    >
      {/* Hero section — slides in from top */}
      <Animated.View style={[
        styles.hero,
        {
          opacity: heroAnim,
          transform: [{
            translateY: heroAnim.interpolate({
              inputRange: [0, 1],
              outputRange: [-30, 0],
            })
          }]
        }
      ]}>
        <View style={styles.heroTop}>
          <View>
            <Text style={styles.greeting}>{getGreeting()},</Text>
            <Text style={styles.name}>{firstName}</Text>
          </View>
          <View style={styles.heroBadge}>
            <Ionicons name="leaf" size={14} color="#4caf50" />
            <Text style={styles.heroBadgeText}>AgriGuard</Text>
          </View>
        </View>

        <Text style={styles.heroSubtitle}>
          Your crops are waiting to be checked.
        </Text>

        {/* Scan button with pulse animation */}
        <Animated.View style={{ transform: [{ scale: scanAnim }] }}>
          <TouchableOpacity
            style={styles.scanButton}
            onPress={pulseScanButton}
            activeOpacity={1}
          >
            <View style={styles.scanButtonIcon}>
              <Ionicons name="scan" size={22} color="#fff" />
            </View>
            <Text style={styles.scanButtonText}>Scan a Crop</Text>
            <Ionicons name="arrow-forward-circle" size={22} color="rgba(255,255,255,0.7)" />
          </TouchableOpacity>
        </Animated.View>
      </Animated.View>

      {/* Cards section — slides in from bottom */}
      <Animated.View style={{
        opacity: cardAnim,
        transform: [{
          translateY: cardAnim.interpolate({
            inputRange: [0, 1],
            outputRange: [40, 0],
          })
        }]
      }}>

        {/* Last scan card */}
        <Text style={styles.sectionTitle}>Last Scan</Text>
        {loading ? (
          <ActivityIndicator color="#2e7d32" style={{ marginVertical: 20 }} />
        ) : recentScan ? (
          <View style={styles.lastScanCard}>
            <View style={[
              styles.scanBar,
              { backgroundColor: recentScan.is_healthy ? '#2e7d32' : '#e53935' }
            ]} />
            <View style={styles.scanInfo}>
              <Text style={styles.scanCrop}>{recentScan.crop}</Text>
              <Text style={styles.scanDisease}>{recentScan.disease}</Text>
              <Text style={styles.scanDate}>
                {new Date(recentScan.createdAt).toLocaleDateString('en-PK', {
                  day: 'numeric', month: 'short', year: 'numeric'
                })}
              </Text>
            </View>
            <View style={styles.scanRight}>
              <View style={[
                styles.scanBadge,
                { backgroundColor: recentScan.is_healthy ? '#e8f5e9' : '#fdecea' }
              ]}>
                <Text style={[
                  styles.scanBadgeText,
                  { color: recentScan.is_healthy ? '#2e7d32' : '#e53935' }
                ]}>
                  {recentScan.is_healthy ? 'Healthy' : 'Diseased'}
                </Text>
              </View>
              <Text style={styles.scanConfidence}>
                {recentScan.confidence.toFixed(1)}%
              </Text>
            </View>
          </View>
        ) : (
          <View style={styles.emptyCard}>
            <Ionicons name="leaf-outline" size={28} color="#ccc" />
            <Text style={styles.emptyText}>No scans yet — scan your first crop above</Text>
          </View>
        )}

        {/* Daily tip */}
        <View style={styles.tipCard}>
          <View style={styles.tipIcon}>
            <Ionicons name="bulb-outline" size={20} color="#f57c00" />
          </View>
          <View style={styles.tipContent}>
            <Text style={styles.tipTitle}>Daily Tip</Text>
            <Text style={styles.tipText}>{getTip()}</Text>
          </View>
        </View>

        {/* Water control shortcut */}
        <TouchableOpacity
          style={styles.waterCard}
          onPress={() => navigation.navigate('WaterControl')}
          activeOpacity={0.85}
        >
          <View style={styles.waterLeft}>
            <View style={styles.waterIcon}>
              <Ionicons name="water" size={22} color="#1565c0" />
            </View>
            <View>
              <Text style={styles.waterTitle}>Irrigation Control</Text>
              <Text style={styles.waterSub}>Manage your water pump remotely</Text>
            </View>
          </View>
          <Ionicons name="chevron-forward" size={20} color="#aaa" />
        </TouchableOpacity>

      </Animated.View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f2f2f7',
  },
  content: {
    paddingBottom: 110,
  },

  // Hero
  hero: {
    backgroundColor: '#1b5e20',
    paddingTop: 60,
    paddingBottom: 32,
    paddingHorizontal: 24,
    marginBottom: 24,
  },
  heroTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  greeting: {
    fontSize: 14,
    color: '#a5d6a7',
  },
  name: {
    fontSize: 30,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 2,
  },
  heroBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.12)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    gap: 5,
  },
  heroBadgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  heroSubtitle: {
    fontSize: 14,
    color: '#81c784',
    marginBottom: 20,
  },
  scanButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2e7d32',
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 16,
    gap: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.15)',
  },
  scanButtonIcon: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.15)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanButtonText: {
    flex: 1,
    fontSize: 16,
    fontWeight: '700',
    color: '#fff',
  },

  // Section title
  sectionTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: '#1a1a1a',
    marginBottom: 12,
    paddingHorizontal: 16,
  },

  // Last scan
  lastScanCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    marginHorizontal: 16,
    marginBottom: 14,
    flexDirection: 'row',
    alignItems: 'center',
    overflow: 'hidden',
    elevation: 2,
  },
  scanBar: {
    width: 5,
    alignSelf: 'stretch',
  },
  scanInfo: {
    flex: 1,
    padding: 16,
  },
  scanCrop: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1a1a1a',
  },
  scanDisease: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  scanDate: {
    fontSize: 11,
    color: '#aaa',
    marginTop: 4,
  },
  scanRight: {
    padding: 16,
    alignItems: 'flex-end',
    gap: 6,
  },
  scanBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 20,
  },
  scanBadgeText: {
    fontSize: 11,
    fontWeight: '600',
  },
  scanConfidence: {
    fontSize: 13,
    fontWeight: '700',
    color: '#888',
  },

  // Empty
  emptyCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    marginHorizontal: 16,
    marginBottom: 14,
    padding: 28,
    alignItems: 'center',
    gap: 10,
    elevation: 1,
  },
  emptyText: {
    fontSize: 13,
    color: '#aaa',
    textAlign: 'center',
  },

  // Tip
  tipCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    marginHorizontal: 16,
    marginBottom: 14,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    elevation: 1,
  },
  tipIcon: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: '#fff3e0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  tipContent: { flex: 1 },
  tipTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#f57c00',
    marginBottom: 3,
  },
  tipText: {
    fontSize: 13,
    color: '#555',
    lineHeight: 18,
  },

  // Water shortcut
  waterCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    marginHorizontal: 16,
    marginBottom: 14,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    elevation: 1,
  },
  waterLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    flex: 1,
  },
  waterIcon: {
    width: 44,
    height: 44,
    borderRadius: 12,
    backgroundColor: '#e3f2fd',
    justifyContent: 'center',
    alignItems: 'center',
  },
  waterTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1a1a1a',
  },
  waterSub: {
    fontSize: 12,
    color: '#888',
    marginTop: 2,
  },
});