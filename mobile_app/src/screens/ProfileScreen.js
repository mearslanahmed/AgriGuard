import React, { useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  Alert, ScrollView, ActivityIndicator, Switch
} from 'react-native';
import { useAuth } from '../context/AuthContext';
import * as SecureStore from 'expo-secure-store';
import { BACKEND_URL } from '../config';

export default function ProfileScreen() {
  const { userInfo, logout } = useAuth();
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [notifications, setNotifications] = useState(true);

  React.useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const token = await SecureStore.getItemAsync('userToken');
      const response = await fetch(`${BACKEND_URL}/api/scans`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        const scans = await response.json();
        const total = scans.length;
        const diseased = scans.filter((s) => !s.is_healthy).length;
        const healthy = scans.filter((s) => s.is_healthy).length;
        setStats({ total, diseased, healthy });
      }
    } catch (err) {
      console.log('Stats fetch failed:', err.message);
    }
  };

  const handleLogout = () => {
    Alert.alert(
      'Log Out',
      'You will be returned to the login screen.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Log Out',
          style: 'destructive',
          onPress: async () => {
            setLoading(true);
            await logout();
          },
        },
      ]
    );
  };

  const getInitials = (name) => {
    if (!name) return '?';
    return name.split(' ').map((n) => n[0]).join('').toUpperCase().slice(0, 2);
  };

  const MenuItem = ({ label, value, onPress, danger, toggle, toggleValue, onToggle }) => (
    <TouchableOpacity
      style={styles.menuItem}
      onPress={onPress}
      disabled={!onPress && !toggle}
      activeOpacity={onPress ? 0.6 : 1}
    >
      <Text style={[styles.menuLabel, danger && styles.dangerText]}>{label}</Text>
      {value && <Text style={styles.menuValue}>{value}</Text>}
      {toggle && (
        <Switch
          value={toggleValue}
          onValueChange={onToggle}
          trackColor={{ false: '#ddd', true: '#a5d6a7' }}
          thumbColor={toggleValue ? '#2e7d32' : '#f4f3f4'}
        />
      )}
      {onPress && !danger && <Text style={styles.chevron}>›</Text>}
    </TouchableOpacity>
  );

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>

      {/* Profile header */}
      <View style={styles.profileHeader}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{getInitials(userInfo?.name)}</Text>
        </View>
        <View style={styles.profileInfo}>
          <Text style={styles.name}>{userInfo?.name}</Text>
          <Text style={styles.email}>{userInfo?.email}</Text>
          <View style={styles.rolePill}>
            <View style={styles.roleDot} />
            <Text style={styles.roleText}>
              {userInfo?.role === 'admin' ? 'Administrator' : 'Farmer Account'}
            </Text>
          </View>
        </View>
      </View>

      {/* Stats */}
      {stats && (
        <View style={styles.statsRow}>
          <View style={styles.statCard}>
            <Text style={styles.statNumber}>{stats.total}</Text>
            <Text style={styles.statLabel}>Total Scans</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statCard}>
            <Text style={[styles.statNumber, { color: '#e53935' }]}>{stats.diseased}</Text>
            <Text style={styles.statLabel}>Diseased</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statCard}>
            <Text style={[styles.statNumber, { color: '#2e7d32' }]}>{stats.healthy}</Text>
            <Text style={styles.statLabel}>Healthy</Text>
          </View>
        </View>
      )}

      {/* Account section */}
      <Text style={styles.sectionLabel}>ACCOUNT</Text>
      <View style={styles.menuCard}>
        <MenuItem label="Full Name" value={userInfo?.name} />
        <View style={styles.menuDivider} />
        <MenuItem label="Email Address" value={userInfo?.email} />
        <View style={styles.menuDivider} />
        <MenuItem label="Account Type" value={userInfo?.role === 'admin' ? 'Administrator' : 'Farmer'} />
      </View>

      {/* Preferences section */}
      <Text style={styles.sectionLabel}>PREFERENCES</Text>
      <View style={styles.menuCard}>
        <MenuItem
          label="Push Notifications"
          toggle
          toggleValue={notifications}
          onToggle={setNotifications}
        />
        <View style={styles.menuDivider} />
        <MenuItem
          label="Language"
          value="English"
          onPress={() => Alert.alert('Coming Soon', 'Urdu language support is coming in a future update.')}
        />
      </View>

      {/* Support section */}
      <Text style={styles.sectionLabel}>SUPPORT</Text>
      <View style={styles.menuCard}>
        <MenuItem
          label="How to Use AgriGuard"
          onPress={() => Alert.alert('Guide', 'Go to Detect tab, take a photo of your crop leaf, and tap Analyze Crop.')}
        />
        <View style={styles.menuDivider} />
        <MenuItem
          label="Supported Crops"
          value="Tomato · Potato · Pepper"
        />
        <View style={styles.menuDivider} />
        <MenuItem
          label="Report a Problem"
          onPress={() => Alert.alert('Report', 'Please email support@agriguard.app with details of the issue.')}
        />
      </View>

      {/* App info */}
      <Text style={styles.sectionLabel}>APP</Text>
      <View style={styles.menuCard}>
        <MenuItem label="Version" value="1.0.0" />
        <View style={styles.menuDivider} />
        <MenuItem label="Model Accuracy" value="~92%" />
        <View style={styles.menuDivider} />
        <MenuItem label="Diseases Covered" value="15 conditions" />
      </View>

      {/* Logout */}
      <TouchableOpacity style={styles.logoutButton} onPress={handleLogout} disabled={loading}>
        {loading
          ? <ActivityIndicator color="#e53935" />
          : <Text style={styles.logoutText}>Log Out</Text>
        }
      </TouchableOpacity>

      <Text style={styles.footer}>AgriGuard · Intelligent Crop Protection</Text>

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f2f2f7', // iOS-style light grey
  },
  content: {
    paddingBottom: 40,
    paddingTop: 35,
  },

  // Profile header
  profileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    padding: 20,
    marginBottom: 8,
    gap: 16,
  },
  avatar: {
    width: 68,
    height: 68,
    borderRadius: 34,
    backgroundColor: '#2e7d32',
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  profileInfo: { flex: 1 },
  name: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1a1a1a',
  },
  email: {
    fontSize: 13,
    color: '#888',
    marginTop: 2,
  },
  rolePill: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 6,
    gap: 5,
  },
  roleDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: '#2e7d32',
  },
  roleText: {
    fontSize: 12,
    color: '#2e7d32',
    fontWeight: '600',
  },

  // Stats bar
  statsRow: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    marginBottom: 28,
    paddingVertical: 16,
  },
  statCard: {
    flex: 1,
    alignItems: 'center',
  },
  statDivider: {
    width: 1,
    backgroundColor: '#eee',
  },
  statNumber: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#1a1a1a',
  },
  statLabel: {
    fontSize: 11,
    color: '#888',
    marginTop: 3,
  },

  // Section labels (iOS settings style)
  sectionLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#888',
    letterSpacing: 0.8,
    marginLeft: 16,
    marginBottom: 6,
    marginTop: 4,
  },

  // Menu cards
  menuCard: {
    backgroundColor: '#fff',
    marginHorizontal: 0,
    marginBottom: 28,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  menuLabel: {
    fontSize: 15,
    color: '#1a1a1a',
    flex: 1,
  },
  menuValue: {
    fontSize: 14,
    color: '#888',
    maxWidth: '55%',
    textAlign: 'right',
  },
  chevron: {
    fontSize: 20,
    color: '#ccc',
    marginLeft: 6,
  },
  menuDivider: {
    height: 1,
    backgroundColor: '#f2f2f7',
    marginLeft: 16,
  },
  dangerText: {
    color: '#e53935',
  },

  // Logout
  logoutButton: {
    marginHorizontal: 16,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#e53935',
    marginBottom: 20,
  },
  logoutText: {
    color: '#e53935',
    fontSize: 15,
    fontWeight: '600',
  },

  footer: {
    textAlign: 'center',
    fontSize: 12,
    color: '#bbb',
    marginBottom: 10,
  },
});