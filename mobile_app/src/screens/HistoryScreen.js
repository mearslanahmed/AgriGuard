import React, { useState, useCallback, useRef } from 'react';
import {
  View, Text, StyleSheet, FlatList,
  TouchableOpacity, ActivityIndicator,
  Alert, RefreshControl, Animated
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import * as SecureStore from 'expo-secure-store';
import { Ionicons } from '@expo/vector-icons';
import { BACKEND_URL } from '../config';

const AnimatedCard = ({ item, index, onDelete }) => {
  const cardAnim = useRef(new Animated.Value(0)).current;
  const deleteScale = useRef(new Animated.Value(1)).current;

  React.useEffect(() => {
    Animated.timing(cardAnim, {
      toValue: 1,
      duration: 350,
      delay: index * 60,
      useNativeDriver: true,
    }).start();
  }, []);

  const pressIn = () =>
    Animated.spring(deleteScale, { toValue: 0.9, useNativeDriver: true }).start();
  const pressOut = () =>
    Animated.spring(deleteScale, { toValue: 1, friction: 4, useNativeDriver: true }).start();

  const getConfidenceColor = (c) => {
    if (c >= 85) return '#2e7d32';
    if (c >= 60) return '#f57c00';
    return '#e53935';
  };

  return (
    <Animated.View style={[styles.card, {
      opacity: cardAnim,
      transform: [{
        translateY: cardAnim.interpolate({
          inputRange: [0, 1], outputRange: [24, 0]
        })
      }]
    }]}>
      {/* Left accent bar */}
      <View style={[
        styles.accentBar,
        { backgroundColor: item.is_healthy ? '#2e7d32' : '#e53935' }
      ]} />

      <View style={styles.cardContent}>
        {/* Top row */}
        <View style={styles.cardTop}>
          <View style={styles.cardTopLeft}>
            <Text style={styles.cropName}>{item.crop}</Text>
            <Text style={styles.diseaseName} numberOfLines={1}>{item.disease}</Text>
          </View>
          <View style={[
            styles.badge,
            { backgroundColor: item.is_healthy ? '#e8f5e9' : '#fdecea' }
          ]}>
            <Ionicons
              name={item.is_healthy ? 'checkmark-circle' : 'warning'}
              size={11}
              color={item.is_healthy ? '#2e7d32' : '#e53935'}
              style={{ marginRight: 4 }}
            />
            <Text style={[
              styles.badgeText,
              { color: item.is_healthy ? '#2e7d32' : '#e53935' }
            ]}>
              {item.is_healthy ? 'Healthy' : 'Diseased'}
            </Text>
          </View>
        </View>

        {/* Bottom row */}
        <View style={styles.cardBottom}>
          <View style={styles.confidenceRow}>
            <Text style={styles.confidenceLabel}>Confidence</Text>
            <Text style={[
              styles.confidenceValue,
              { color: getConfidenceColor(item.confidence) }
            ]}>
              {item.confidence.toFixed(1)}%
            </Text>
          </View>

          <View style={styles.bottomRight}>
            <Text style={styles.date}>
              {new Date(item.createdAt).toLocaleDateString('en-PK', {
                day: 'numeric', month: 'short', year: 'numeric'
              })}
            </Text>
            <Animated.View style={{ transform: [{ scale: deleteScale }] }}>
              <TouchableOpacity
                style={styles.deleteBtn}
                onPress={() => onDelete(item._id)}
                onPressIn={pressIn}
                onPressOut={pressOut}
                activeOpacity={1}
              >
                <Ionicons name="trash-outline" size={14} color="#e53935" />
              </TouchableOpacity>
            </Animated.View>
          </View>
        </View>
      </View>
    </Animated.View>
  );
};

export default function HistoryScreen({ onOpen }) {
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const headerAnim = useRef(new Animated.Value(0)).current;

  useFocusEffect(
    useCallback(() => {
      if (onOpen) onOpen();
      fetchScans();

      headerAnim.setValue(0);
      Animated.timing(headerAnim, {
        toValue: 1, duration: 400,
        useNativeDriver: true,
      }).start();
    }, [onOpen])
  );

  const fetchScans = async () => {
    try {
      const token = await SecureStore.getItemAsync('userToken');
      const response = await fetch(`${BACKEND_URL}/api/scans`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await response.json();
      if (response.ok) setScans(data);
    } catch (err) {
      console.log('History fetch error:', err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleDelete = (id) => {
    Alert.alert(
      'Delete Scan',
      'This record will be permanently removed.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Delete', style: 'destructive', onPress: () => deleteScan(id) },
      ]
    );
  };

  const deleteScan = async (id) => {
    try {
      const token = await SecureStore.getItemAsync('userToken');
      const response = await fetch(`${BACKEND_URL}/api/scans/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) setScans((prev) => prev.filter((s) => s._id !== id));
    } catch (err) {
      Alert.alert('Error', 'Could not delete scan.');
    }
  };

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#2e7d32" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <Animated.View style={[styles.header, {
        opacity: headerAnim,
        transform: [{
          translateY: headerAnim.interpolate({
            inputRange: [0, 1], outputRange: [-10, 0]
          })
        }]
      }]}>
        <Text style={styles.title}>Scan History</Text>
        {scans.length > 0 && (
          <Text style={styles.count}>{scans.length} record{scans.length !== 1 ? 's' : ''}</Text>
        )}
      </Animated.View>

      <FlatList
        data={scans}
        keyExtractor={(item) => item._id}
        renderItem={({ item, index }) => (
          <AnimatedCard item={item} index={index} onDelete={handleDelete} />
        )}
        contentContainerStyle={[
          styles.list,
          scans.length === 0 && styles.centered
        ]}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={() => { setRefreshing(true); fetchScans(); }}
            colors={['#2e7d32']}
          />
        }
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <View style={styles.emptyIconBox}>
              <Ionicons name="leaf-outline" size={32} color="#2e7d32" />
            </View>
            <Text style={styles.emptyTitle}>No scans yet</Text>
            <Text style={styles.emptyHint}>Go to Detect to analyze your first crop.</Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f7faf7',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 35,
    paddingBottom: 12,
  },
  title: {
    fontSize: 26,
    fontWeight: '700',
    color: '#1b5e20',
    letterSpacing: -0.5,
  },
  count: {
    fontSize: 13,
    color: '#999',
    fontWeight: '500',
  },
  list: {
    paddingHorizontal: 16,
    paddingBottom: 110,
  },

  // Card
  card: {
    backgroundColor: '#fff',
    borderRadius: 14,
    marginBottom: 10,
    flexDirection: 'row',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#e8f0e8',
  },
  accentBar: {
    width: 4,
  },
  cardContent: {
    flex: 1,
    padding: 14,
    gap: 10,
  },
  cardTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  cardTopLeft: {
    flex: 1,
    marginRight: 10,
  },
  cropName: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1a1a1a',
  },
  diseaseName: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 20,
  },
  badgeText: {
    fontSize: 11,
    fontWeight: '600',
  },
  cardBottom: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  confidenceLabel: {
    fontSize: 11,
    color: '#bbb',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  confidenceValue: {
    fontSize: 13,
    fontWeight: '700',
  },
  bottomRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  date: {
    fontSize: 11,
    color: '#bbb',
  },
  deleteBtn: {
    width: 30,
    height: 30,
    borderRadius: 8,
    backgroundColor: '#fdecea',
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Empty state
  emptyState: {
    alignItems: 'center',
    gap: 10,
  },
  emptyIconBox: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#e8f5e9',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 4,
  },
  emptyTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#555',
  },
  emptyHint: {
    fontSize: 13,
    color: '#aaa',
  },
});