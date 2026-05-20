import React, { useState, useEffect } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  ScrollView, ActivityIndicator, Alert
} from 'react-native';
import * as SecureStore from 'expo-secure-store';
import { BACKEND_URL } from '../config';

export default function WaterControlScreen({ navigation }) {
  const [pumpOn, setPumpOn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [statusLoading, setStatusLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('checking'); // checking | online | offline

  useEffect(() => {
    fetchPumpStatus();
  }, []);

  const fetchPumpStatus = async () => {
    try {
      const token = await SecureStore.getItemAsync('userToken');
      const response = await fetch(`${BACKEND_URL}/api/water/status`, {
        headers: { Authorization: `Bearer ${token}` },
        // Short timeout — ESP32 might be slow to respond
        signal: AbortSignal.timeout(5000),
      });

      if (response.ok) {
        const data = await response.json();
        setPumpOn(data.pump_on);
        setLastUpdated(new Date());
        setConnectionStatus('online');
      } else {
        setConnectionStatus('offline');
      }
    } catch (err) {
      // ESP32 not reachable — show offline state gracefully
      setConnectionStatus('offline');
      console.log('Pump status fetch failed:', err.message);
    } finally {
      setStatusLoading(false);
    }
  };

  const togglePump = async () => {
    if (connectionStatus === 'offline') {
      Alert.alert(
        'Device Offline',
        'Cannot reach the irrigation controller. Make sure the ESP32 device is powered on and connected to the same network.'
      );
      return;
    }

    const action = pumpOn ? 'turn OFF' : 'turn ON';
    Alert.alert(
      'Confirm Action',
      `Are you sure you want to ${action} the water pump?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Confirm',
          onPress: async () => {
            setLoading(true);
            try {
              const token = await SecureStore.getItemAsync('userToken');
              const response = await fetch(`${BACKEND_URL}/api/water/control`, {
                method: 'POST',
                headers: {
                  Authorization: `Bearer ${token}`,
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({ pump_on: !pumpOn }),
              });

              if (response.ok) {
                setPumpOn(!pumpOn);
                setLastUpdated(new Date());
              } else {
                Alert.alert('Error', 'Failed to send command to device.');
              }
            } catch (err) {
              Alert.alert('Error', 'Could not reach the irrigation controller.');
            } finally {
              setLoading(false);
            }
          },
        },
      ]
    );
  };

  const formatTime = (date) => {
    if (!date) return 'Never';
    return date.toLocaleTimeString('en-PK', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>

      {/* Header */}
      <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
        <Text style={styles.backText}>← Back</Text>
      </TouchableOpacity>
      <Text style={styles.title}>Water Management</Text>
      <Text style={styles.subtitle}>Control your irrigation pump remotely</Text>

      {/* Connection status bar */}
      <View style={[
        styles.statusBar,
        { backgroundColor: connectionStatus === 'online' ? '#e8f5e9' : '#fdecea' }
      ]}>
        <View style={[
          styles.statusDot,
          { backgroundColor: connectionStatus === 'online' ? '#2e7d32' : '#e53935' }
        ]} />
        <Text style={[
          styles.statusText,
          { color: connectionStatus === 'online' ? '#2e7d32' : '#e53935' }
        ]}>
          {connectionStatus === 'checking' && 'Connecting to device...'}
          {connectionStatus === 'online' && 'ESP32 Controller Online'}
          {connectionStatus === 'offline' && 'Controller Offline — Check Device'}
        </Text>
        <TouchableOpacity onPress={fetchPumpStatus}>
          <Text style={styles.refreshText}>Refresh</Text>
        </TouchableOpacity>
      </View>

      {/* Main pump control */}
      {statusLoading ? (
        <ActivityIndicator size="large" color="#2e7d32" style={{ marginTop: 60 }} />
      ) : (
        <>
          {/* Pump status card */}
          <View style={styles.pumpCard}>
            <Text style={styles.pumpLabel}>Irrigation Pump</Text>

            {/* Big status indicator */}
            <View style={[
              styles.pumpIndicator,
              { backgroundColor: pumpOn ? '#e8f5e9' : '#f5f5f5' }
            ]}>
              <View style={[
                styles.pumpDot,
                { backgroundColor: pumpOn ? '#2e7d32' : '#bbb' }
              ]} />
              <Text style={[
                styles.pumpStatus,
                { color: pumpOn ? '#2e7d32' : '#888' }
              ]}>
                {pumpOn ? 'RUNNING' : 'STOPPED'}
              </Text>
            </View>

            {/* Toggle button */}
            <TouchableOpacity
              style={[
                styles.toggleButton,
                { backgroundColor: pumpOn ? '#e53935' : '#2e7d32' },
                loading && styles.buttonDisabled,
              ]}
              onPress={togglePump}
              disabled={loading}
            >
              {loading
                ? <ActivityIndicator color="#fff" />
                : <Text style={styles.toggleText}>
                    {pumpOn ? 'Turn Off Pump' : 'Turn On Pump'}
                  </Text>
              }
            </TouchableOpacity>

            <Text style={styles.lastUpdated}>
              Last updated: {formatTime(lastUpdated)}
            </Text>
          </View>

          {/* Info cards */}
          <View style={styles.infoRow}>
            <View style={styles.infoCard}>
              <Text style={styles.infoTitle}>Device</Text>
              <Text style={styles.infoValue}>ESP32-CAM</Text>
            </View>
            <View style={styles.infoCard}>
              <Text style={styles.infoTitle}>Protocol</Text>
              <Text style={styles.infoValue}>Wi-Fi / HTTP</Text>
            </View>
          </View>

          {/* Safety notice */}
          <View style={styles.safetyCard}>
            <Text style={styles.safetyTitle}>Safety Notice</Text>
            <Text style={styles.safetyText}>
              Always confirm field conditions before activating the pump remotely.
              Do not run the pump when the water tank is empty.
            </Text>
          </View>
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f2f2f7',
  },
  content: {
    padding: 24,
    paddingBottom: 40,
  },
  backButton: {
    marginTop: 10,
    marginBottom: 10,
  },
  backText: {
    color: '#2e7d32',
    fontSize: 15,
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#1a1a1a',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
    marginBottom: 20,
  },

  // Connection status
  statusBar: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 10,
    marginBottom: 24,
    gap: 8,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  statusText: {
    flex: 1,
    fontSize: 13,
    fontWeight: '500',
  },
  refreshText: {
    fontSize: 13,
    color: '#2e7d32',
    fontWeight: '600',
  },

  // Pump card
  pumpCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    marginBottom: 16,
    elevation: 2,
  },
  pumpLabel: {
    fontSize: 13,
    color: '#888',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 20,
  },
  pumpIndicator: {
    width: 140,
    height: 140,
    borderRadius: 70,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 28,
  },
  pumpDot: {
    width: 20,
    height: 20,
    borderRadius: 10,
    marginBottom: 8,
  },
  pumpStatus: {
    fontSize: 18,
    fontWeight: 'bold',
    letterSpacing: 2,
  },
  toggleButton: {
    width: '100%',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 12,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  toggleText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  lastUpdated: {
    fontSize: 12,
    color: '#aaa',
    marginTop: 4,
  },

  // Info row
  infoRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  infoCard: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    elevation: 1,
  },
  infoTitle: {
    fontSize: 11,
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 6,
  },
  infoValue: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1a1a1a',
  },

  // Safety
  safetyCard: {
    backgroundColor: '#fff3e0',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#f57c00',
  },
  safetyTitle: {
    fontWeight: 'bold',
    color: '#e65100',
    marginBottom: 6,
    fontSize: 14,
  },
  safetyText: {
    color: '#555',
    fontSize: 13,
    lineHeight: 20,
  },
});