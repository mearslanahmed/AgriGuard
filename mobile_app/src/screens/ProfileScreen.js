import React, { useState, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  Alert, ScrollView, ActivityIndicator, Switch,
  Image, Animated, Easing
} from 'react-native';
import { useAuth } from '../context/AuthContext';
import * as SecureStore from 'expo-secure-store';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { BACKEND_URL } from '../config';

export default function ProfileScreen({ navigation }) {
  const { userInfo, logout } = useAuth();
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [notifications, setNotifications] = useState(true);
  const [profilePic, setProfilePic] = useState(null);

  // Animations
  const headerAnim = useRef(new Animated.Value(0)).current;
  const statsAnim = useRef(new Animated.Value(0)).current;
  const sectionsAnim = useRef(new Animated.Value(0)).current;
  const avatarScale = useRef(new Animated.Value(1)).current;

  React.useEffect(() => {
  const unsubscribe = navigation.addListener('focus', () => {
    // reset all anims
    headerAnim.setValue(0);
    statsAnim.setValue(0);
    sectionsAnim.setValue(0);

    Animated.stagger(100, [
      Animated.timing(headerAnim, {
        toValue: 1, duration: 400,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(statsAnim, {
        toValue: 1, duration: 380,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(sectionsAnim, {
        toValue: 1, duration: 380,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
    ]).start();
  });

  fetchStats();
  return unsubscribe;
}, [navigation]);

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

  const handleChangePic = () => {
    Alert.alert('Profile Photo', 'Choose an option', [
      {
        text: 'Take Photo',
        onPress: async () => {
          const permission = await ImagePicker.requestCameraPermissionsAsync();
          if (!permission.granted) {
            Alert.alert('Permission required', 'Please allow camera access.');
            return;
          }
          const result = await ImagePicker.launchCameraAsync({ quality: 0.8, allowsEditing: true, aspect: [1, 1] });
          if (!result.canceled) setProfilePic(result.assets[0].uri);
        },
      },
      {
        text: 'Choose from Gallery',
        onPress: async () => {
          const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
          if (!permission.granted) {
            Alert.alert('Permission required', 'Please allow gallery access.');
            return;
          }
          const result = await ImagePicker.launchImageLibraryAsync({
            quality: 0.8, allowsEditing: true, aspect: [1, 1],
          });
          if (!result.canceled) setProfilePic(result.assets[0].uri);
        },
      },
      profilePic && {
        text: 'Remove Photo',
        style: 'destructive',
        onPress: () => setProfilePic(null),
      },
      { text: 'Cancel', style: 'cancel' },
    ].filter(Boolean));
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

  const pressIn = () =>
    Animated.spring(avatarScale, { toValue: 0.94, useNativeDriver: true }).start();
  const pressOut = () =>
    Animated.spring(avatarScale, { toValue: 1, friction: 4, useNativeDriver: true }).start();

  const animStyle = (anim, slide = 16) => ({
    opacity: anim,
    transform: [{ translateY: anim.interpolate({ inputRange: [0, 1], outputRange: [slide, 0] }) }],
  });

  const MenuItem = ({ icon, label, value, onPress, danger, toggle, toggleValue, onToggle }) => (
    <TouchableOpacity
      style={styles.menuItem}
      onPress={onPress}
      disabled={!onPress && !toggle}
      activeOpacity={onPress ? 0.6 : 1}
    >
      <View style={[styles.menuIconBox, danger && styles.menuIconBoxDanger]}>
        <Ionicons name={icon} size={16} color={danger ? '#e53935' : '#2e7d32'} />
      </View>
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
      {onPress && !danger && (
        <Ionicons name="chevron-forward" size={16} color="#ccc" />
      )}
    </TouchableOpacity>
  );

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Profile header card */}
      <Animated.View style={[styles.headerCard, animStyle(headerAnim, -10)]}>
        {/* Avatar */}
        <Animated.View style={{ transform: [{ scale: avatarScale }] }}>
          <TouchableOpacity
            style={styles.avatarWrapper}
            onPress={handleChangePic}
            onPressIn={pressIn}
            onPressOut={pressOut}
            activeOpacity={1}
          >
            {profilePic ? (
              <Image source={{ uri: profilePic }} style={styles.avatarImage} />
            ) : (
              <View style={styles.avatarFallback}>
                <Text style={styles.avatarText}>{getInitials(userInfo?.name)}</Text>
              </View>
            )}
            <View style={styles.cameraBadge}>
              <Ionicons name="camera" size={11} color="#fff" />
            </View>
          </TouchableOpacity>
        </Animated.View>

        {/* Name + email */}
        <Text style={styles.name}>{userInfo?.name}</Text>
        <Text style={styles.email}>{userInfo?.email}</Text>

        <View style={styles.rolePill}>
          <View style={styles.roleDot} />
          <Text style={styles.roleText}>
            {userInfo?.role === 'admin' ? 'Administrator' : 'Farmer Account'}
          </Text>
        </View>

        {/* Edit profile */}
        <TouchableOpacity
          style={styles.editBtn}
          onPress={() => navigation.navigate('EditProfile')}
          activeOpacity={0.75}
        >
          <Ionicons name="pencil-outline" size={14} color="#2e7d32" style={{ marginRight: 6 }} />
          <Text style={styles.editBtnText}>Edit Profile</Text>
        </TouchableOpacity>
      </Animated.View>

      {/* Stats */}
      {stats && (
        <Animated.View style={[styles.statsRow, animStyle(statsAnim)]}>
          {[
            { label: 'Total Scans', value: stats.total, color: '#1a1a1a' },
            { label: 'Diseased', value: stats.diseased, color: '#e53935' },
            { label: 'Healthy', value: stats.healthy, color: '#2e7d32' },
          ].map((s, i, arr) => (
            <React.Fragment key={s.label}>
              <View style={styles.statCell}>
                <Text style={[styles.statNumber, { color: s.color }]}>{s.value}</Text>
                <Text style={styles.statLabel}>{s.label}</Text>
              </View>
              {i < arr.length - 1 && <View style={styles.statDivider} />}
            </React.Fragment>
          ))}
        </Animated.View>
      )}

      {/* Sections */}
      <Animated.View style={animStyle(sectionsAnim)}>

        <Text style={styles.sectionLabel}>PREFERENCES</Text>
        <View style={styles.menuCard}>
          <MenuItem
            icon="notifications-outline"
            label="Push Notifications"
            toggle
            toggleValue={notifications}
            onToggle={setNotifications}
          />
          <View style={styles.menuDivider} />
          <MenuItem
            icon="language-outline"
            label="Language"
            value="English"
            onPress={() => Alert.alert('Coming Soon', 'Urdu language support is coming in a future update.')}
          />
        </View>

        <Text style={styles.sectionLabel}>SUPPORT</Text>
        <View style={styles.menuCard}>
          <MenuItem
            icon="help-circle-outline"
            label="How to Use AgriGuard"
            onPress={() => Alert.alert('Guide', 'Go to Detect tab, take a photo of your crop leaf, and tap Analyze Crop.')}
          />
          <View style={styles.menuDivider} />
          <MenuItem
            icon="leaf-outline"
            label="Supported Crops"
            value="Tomato · Potato · Pepper"
          />
          <View style={styles.menuDivider} />
          <MenuItem
            icon="bug-outline"
            label="Report a Problem"
            onPress={() => Alert.alert('Report', 'Please email support@agriguard.app with details of the issue.')}
          />
        </View>

        <Text style={styles.sectionLabel}>APP</Text>
        <View style={styles.menuCard}>
          <MenuItem
            icon="information-circle-outline"
            label="Version"
            value="1.0.0"
          />
        </View>

        {/* Logout */}
        <TouchableOpacity
          style={styles.logoutButton}
          onPress={handleLogout}
          disabled={loading}
          activeOpacity={0.75}
        >
          {loading
            ? <ActivityIndicator color="#e53935" />
            : (
              <>
                <Ionicons name="log-out-outline" size={18} color="#e53935" style={{ marginRight: 8 }} />
                <Text style={styles.logoutText}>Log Out</Text>
              </>
            )
          }
        </TouchableOpacity>

        <Text style={styles.footer}>AgriGuard · Intelligent Crop Protection</Text>

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
    paddingTop: 35,
  },

  // Header card
  headerCard: {
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginBottom: 14,
    borderRadius: 20,
    padding: 24,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#e8f0e8',
  },

  // Avatar
  avatarWrapper: {
    position: 'relative',
    marginBottom: 14,
  },
  avatarImage: {
    width: 84,
    height: 84,
    borderRadius: 42,
    borderWidth: 3,
    borderColor: '#e8f5e9',
  },
  avatarFallback: {
    width: 84,
    height: 84,
    borderRadius: 42,
    backgroundColor: '#2e7d32',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#e8f5e9',
  },
  avatarText: {
    fontSize: 28,
    fontWeight: '700',
    color: '#fff',
  },
  cameraBadge: {
    position: 'absolute',
    bottom: 2,
    right: 2,
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#2e7d32',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#fff',
  },

  name: {
    fontSize: 20,
    fontWeight: '700',
    color: '#1a1a1a',
    marginBottom: 3,
  },
  email: {
    fontSize: 13,
    color: '#999',
    marginBottom: 10,
  },
  rolePill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#e8f5e9',
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 20,
    gap: 6,
    marginBottom: 16,
  },
  roleDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#2e7d32',
  },
  roleText: {
    fontSize: 12,
    color: '#2e7d32',
    fontWeight: '600',
  },
  editBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#2e7d32',
    borderRadius: 10,
    paddingHorizontal: 20,
    paddingVertical: 9,
  },
  editBtnText: {
    color: '#2e7d32',
    fontWeight: '600',
    fontSize: 14,
  },

  // Stats
  statsRow: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    marginHorizontal: 16,
    borderRadius: 16,
    paddingVertical: 18,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: '#e8f0e8',
  },
  statCell: {
    flex: 1,
    alignItems: 'center',
  },
  statDivider: {
    width: 1,
    backgroundColor: '#eee',
  },
  statNumber: {
    fontSize: 22,
    fontWeight: '700',
  },
  statLabel: {
    fontSize: 11,
    color: '#999',
    marginTop: 3,
  },

  // Section labels
  sectionLabel: {
    fontSize: 11,
    fontWeight: '600',
    color: '#999',
    letterSpacing: 0.8,
    marginLeft: 16,
    marginBottom: 6,
    marginTop: 4,
  },

  // Menu
  menuCard: {
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginBottom: 24,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#e8f0e8',
    overflow: 'hidden',
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 14,
    gap: 12,
  },
  menuIconBox: {
    width: 30,
    height: 30,
    borderRadius: 8,
    backgroundColor: '#e8f5e9',
    justifyContent: 'center',
    alignItems: 'center',
  },
  menuIconBoxDanger: {
    backgroundColor: '#fdecea',
  },
  menuLabel: {
    flex: 1,
    fontSize: 15,
    color: '#1a1a1a',
  },
  menuValue: {
    fontSize: 13,
    color: '#999',
    maxWidth: '45%',
    textAlign: 'right',
  },
  menuDivider: {
    height: 1,
    backgroundColor: '#f2f2f7',
    marginLeft: 58,
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
    flexDirection: 'row',
    justifyContent: 'center',
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