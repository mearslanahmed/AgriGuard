import React, { useRef } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  ActivityIndicator, Animated
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

import { useAuth } from '../context/AuthContext';

import LoginScreen from '../screens/LoginScreen';
import RegisterScreen from '../screens/RegisterScreen';
import HomeScreen from '../screens/HomeScreen';
import DetectScreen from '../screens/DetectScreen';
import HistoryScreen from '../screens/HistoryScreen';
import ProfileScreen from '../screens/ProfileScreen';
import ResultScreen from '../screens/ResultScreen';
import WaterControlScreen from '../screens/WaterControlScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

const TABS = [
  { name: 'Home', icon: 'home-outline', iconActive: 'home', label: 'Home' },
  { name: 'Detect', icon: 'scan-outline', iconActive: 'scan', label: 'Detect' },
  { name: 'History', icon: 'time-outline', iconActive: 'time', label: 'History' },
  { name: 'Profile', icon: 'person-outline', iconActive: 'person', label: 'Profile' },
];

function CustomTabBar({ state, navigation, unreadScans = 0 }) {
  const scales = useRef(TABS.map(() => new Animated.Value(1))).current;

  const onPress = (index, routeName, isFocused) => {
    // Haptic feedback on every tab press
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

    Animated.sequence([
      Animated.timing(scales[index], {
        toValue: 0.85,
        duration: 80,
        useNativeDriver: true,
      }),
      Animated.timing(scales[index], {
        toValue: 1,
        duration: 120,
        useNativeDriver: true,
      }),
    ]).start();

    const event = navigation.emit({
      type: 'tabPress',
      target: state.routes[index].key,
      canPreventDefault: true,
    });

    if (!isFocused && !event.defaultPrevented) {
      navigation.navigate(routeName);
    }
  };

  return (
    <View style={styles.navWrapper}>
      <View style={styles.navContainer}>
        {TABS.map((tab, index) => {
          const isFocused = state.index === index;
          // Show badge on History tab when there are scans
          const showBadge = tab.name === 'History' && unreadScans > 0;

          return (
            <Animated.View
              key={tab.name}
              style={[
                styles.tabItem,
                { transform: [{ scale: scales[index] }] }
              ]}
            >
              <TouchableOpacity
                style={styles.tabTouchable}
                onPress={() => onPress(index, tab.name, isFocused)}
                activeOpacity={0.7}
              >
                {isFocused && <View style={styles.activePill} />}

                <View style={styles.iconWrapper}>
                  <Ionicons
                    name={isFocused ? tab.iconActive : tab.icon}
                    size={22}
                    color={isFocused ? '#2e7d32' : '#aaa'}
                  />
                  {/* Badge dot on History */}
                  {showBadge && (
                    <View style={styles.badge}>
                      <Text style={styles.badgeText}>
                        {unreadScans > 9 ? '9+' : unreadScans}
                      </Text>
                    </View>
                  )}
                </View>

                <Text style={[
                  styles.tabLabel,
                  {
                    color: isFocused ? '#2e7d32' : '#aaa',
                    // Active label is bold and slightly larger
                    fontWeight: isFocused ? '700' : '500',
                    fontSize: isFocused ? 12 : 11,
                  }
                ]}>
                  {tab.label}
                </Text>
              </TouchableOpacity>
            </Animated.View>
          );
        })}
      </View>
    </View>
  );
}

function MainTabs() {
  const [unreadCount, setUnreadCount] = React.useState(0);

  // Called by DetectScreen after every successful scan
  const incrementUnread = () => setUnreadCount((prev) => prev + 1);

  // Called when user opens History tab — clears badge
  const clearUnread = () => setUnreadCount(0);

  return (
    <Tab.Navigator
      tabBar={(props) => <CustomTabBar {...props} unreadScans={unreadCount} />}
      screenOptions={{ headerShown: false }}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen
        name="Detect"
        children={(props) => <DetectScreen {...props} onScanComplete={incrementUnread} />}
      />
      <Tab.Screen
        name="History"
        children={() => <HistoryScreen onOpen={clearUnread} />}
      />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  );
}

export default function AppNavigator() {
  const { userToken, isLoading } = useAuth();

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" color="#2e7d32" />
      </View>
    );
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {userToken ? (
          <>
            <Stack.Screen name="MainTabs" component={MainTabs} />
            <Stack.Screen name="Result" component={ResultScreen} />
            <Stack.Screen name="WaterControl" component={WaterControlScreen} />
          </>
        ) : (
          <>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="Register" component={RegisterScreen} />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  navWrapper: {
    position: 'absolute',
    bottom: 24,
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  navContainer: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 30,
    paddingVertical: 10,
    paddingHorizontal: 12,
    width: '100%',
    justifyContent: 'space-around',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.12,
    shadowRadius: 16,
    elevation: 12,
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
  },
  tabTouchable: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 20,
    position: 'relative',
    minWidth: 60,
  },
  activePill: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: '#e8f5e9',
    borderRadius: 20,
  },
  iconWrapper: {
    position: 'relative',
  },
  badge: {
    position: 'absolute',
    top: -4,
    right: -8,
    backgroundColor: '#e53935',
    borderRadius: 8,
    minWidth: 16,
    height: 16,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 3,
  },
  badgeText: {
    color: '#fff',
    fontSize: 9,
    fontWeight: 'bold',
  },
  tabLabel: {
    marginTop: 3,
  },
});