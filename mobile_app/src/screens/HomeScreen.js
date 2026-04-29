import React, { useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  ScrollView, ActivityIndicator
} from 'react-native';
import { useAuth } from '../context/AuthContext';
import * as SecureStore from 'expo-secure-store';
import { BACKEND_URL } from '../config';

export default function HomeScreen({ navigation }) {
  const { userInfo } = useAuth();
  const [recentScans, setRecentScans] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch last 3 scans to show on dashboard
  useEffect(() => {
    fetchRecentScans();
  }, []);

  const fetchRecentScans = async () => {
    try {
      const token = await SecureStore.getItemAsync('userToken');
      const response = await fetch(`${BACKEND_URL}/api/scans?limit=3`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        const data = await response.json();
        setRecentScans(data);
      }
    } catch (err) {
      // Silently fail — home screen still usable without recent scans
      console.log('Could not fetch recent scans:', err.message);
    } finally {
      setLoading(false);
    }
  };

  // First name only for the greeting
  const firstName = userInfo?.name?.split(' ')[0] || 'Farmer';

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>

      {/* Greeting */}
      <View style={styles.header}>
        <Text style={styles.greeting}>Hello, {firstName}</Text>
        <Text style={styles.subGreeting}>Monitor your crops and stay ahead of disease.</Text>
      </View>

      {/* Quick action — main CTA of the whole app */}
      <TouchableOpacity
        style={styles.scanButton}
        onPress={() => navigation.navigate('Detect')}
      >
        <Text style={styles.scanButtonText}>Scan a Crop</Text>
        <Text style={styles.scanButtonSub}>Detect disease instantly</Text>
      </TouchableOpacity>

      {/* Stats row */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>15</Text>
          <Text style={styles.statLabel}>Diseases Covered</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>3</Text>
          <Text style={styles.statLabel}>Crops Supported</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statNumber}>92%</Text>
          <Text style={styles.statLabel}>Model Accuracy</Text>
        </View>
      </View>

      {/* Recent scans section */}
      <Text style={styles.sectionTitle}>Recent Scans</Text>

      {loading ? (
        <ActivityIndicator color="#2e7d32" style={{ marginTop: 20 }} />
      ) : recentScans.length === 0 ? (
        <View style={styles.emptyCard}>
          <Text style={styles.emptyText}>No scans yet. Scan your first crop above.</Text>
        </View>
      ) : (
        recentScans.map((scan, index) => (
          <View key={index} style={styles.scanCard}>
            <View style={styles.scanCardLeft}>
              <Text style={styles.scanCrop}>{scan.crop}</Text>
              <Text style={styles.scanDisease}>{scan.disease}</Text>
              <Text style={styles.scanDate}>
                {new Date(scan.createdAt).toLocaleDateString()}
              </Text>
            </View>
            <View style={[
              styles.scanBadge,
              { backgroundColor: scan.is_healthy ? '#e8f5e9' : '#fdecea' }
            ]}>
              <Text style={[
                styles.scanBadgeText,
                { color: scan.is_healthy ? '#2e7d32' : '#e53935' }
              ]}>
                {scan.is_healthy ? 'Healthy' : 'Diseased'}
              </Text>
            </View>
          </View>
        ))
      )}

      {/* Water control shortcut */}
      <TouchableOpacity
        style={styles.waterButton}
        onPress={() => navigation.navigate('WaterControl')}
      >
        <Text style={styles.waterButtonText}>Water Management</Text>
        <Text style={styles.waterButtonSub}>Control irrigation pump</Text>
      </TouchableOpacity>

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  content: {
    padding: 24,
    paddingBottom: 40,
    paddingBottom: 100, // clears floating nav
  },
  header: {
    marginBottom: 24,
    marginTop: 10,
  },
  greeting: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#1b5e20',
  },
  subGreeting: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },

  // Main scan CTA
  scanButton: {
    backgroundColor: '#2e7d32',
    borderRadius: 16,
    padding: 24,
    marginBottom: 20,
    elevation: 3,
  },
  scanButtonText: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#fff',
  },
  scanButtonSub: {
    fontSize: 13,
    color: '#a5d6a7',
    marginTop: 4,
  },

  // Stats
  statsRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 28,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 14,
    alignItems: 'center',
    elevation: 1,
  },
  statNumber: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2e7d32',
  },
  statLabel: {
    fontSize: 11,
    color: '#888',
    textAlign: 'center',
    marginTop: 4,
  },

  // Recent scans
  sectionTitle: {
    fontSize: 17,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  emptyCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    marginBottom: 20,
  },
  emptyText: {
    color: '#aaa',
    fontSize: 14,
  },
  scanCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    elevation: 1,
  },
  scanCardLeft: { flex: 1 },
  scanCrop: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#333',
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
  scanBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  scanBadgeText: {
    fontSize: 12,
    fontWeight: '600',
  },

  // Water control shortcut
  waterButton: {
    backgroundColor: '#1565c0',
    borderRadius: 16,
    padding: 20,
    marginTop: 10,
    elevation: 2,
  },
  waterButtonText: {
    fontSize: 17,
    fontWeight: 'bold',
    color: '#fff',
  },
  waterButtonSub: {
    fontSize: 13,
    color: '#90caf9',
    marginTop: 4,
  },
});