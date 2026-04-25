import React, { useState, useCallback } from 'react';
import {
  View, Text, StyleSheet, FlatList,
  TouchableOpacity, ActivityIndicator,
  Alert, RefreshControl
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import * as SecureStore from 'expo-secure-store';
import { BACKEND_URL } from '../config';

export default function HistoryScreen() {
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Reload scans every time farmer navigates to this tab
  useFocusEffect(
    useCallback(() => {
      fetchScans();
    }, [])
  );

  const fetchScans = async () => {
    try {
      const token = await SecureStore.getItemAsync('userToken');
      const response = await fetch(`${BACKEND_URL}/api/scans`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        const data = await response.json();
        setScans(data);
      }
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
      'Are you sure you want to delete this record?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => deleteScan(id),
        },
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
      if (response.ok) {
        // Remove from local state without refetching
        setScans((prev) => prev.filter((s) => s._id !== id));
      }
    } catch (err) {
      Alert.alert('Error', 'Could not delete scan.');
    }
  };

  const renderItem = ({ item }) => (
    <View style={styles.card}>
      <View style={styles.cardLeft}>
        <Text style={styles.crop}>{item.crop}</Text>
        <Text style={styles.disease}>{item.disease}</Text>
        <Text style={styles.confidence}>Confidence: {item.confidence.toFixed(1)}%</Text>
        <Text style={styles.date}>
          {new Date(item.createdAt).toLocaleDateString('en-PK', {
            day: 'numeric', month: 'short', year: 'numeric'
          })}
        </Text>
      </View>

      <View style={styles.cardRight}>
        <View style={[
          styles.badge,
          { backgroundColor: item.is_healthy ? '#e8f5e9' : '#fdecea' }
        ]}>
          <Text style={[
            styles.badgeText,
            { color: item.is_healthy ? '#2e7d32' : '#e53935' }
          ]}>
            {item.is_healthy ? 'Healthy' : 'Diseased'}
          </Text>
        </View>

        <TouchableOpacity
          style={styles.deleteButton}
          onPress={() => handleDelete(item._id)}
        >
          <Text style={styles.deleteText}>Delete</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#2e7d32" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Scan History</Text>

      <FlatList
        data={scans}
        keyExtractor={(item) => item._id}
        renderItem={renderItem}
        contentContainerStyle={scans.length === 0 && styles.centered}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={() => { setRefreshing(true); fetchScans(); }}
            colors={['#2e7d32']}
          />
        }
        ListEmptyComponent={
          <Text style={styles.emptyText}>No scans yet. Go detect a crop disease.</Text>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2e7d32',
    marginBottom: 16,
    marginTop: 10,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    elevation: 1,
  },
  cardLeft: { flex: 1 },
  crop: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  disease: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  confidence: {
    fontSize: 12,
    color: '#888',
    marginTop: 2,
  },
  date: {
    fontSize: 11,
    color: '#aaa',
    marginTop: 4,
  },
  cardRight: {
    alignItems: 'flex-end',
    gap: 8,
  },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
  },
  badgeText: {
    fontSize: 11,
    fontWeight: '600',
  },
  deleteButton: {
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  deleteText: {
    color: '#e53935',
    fontSize: 12,
    fontWeight: '500',
  },
  emptyText: {
    color: '#aaa',
    fontSize: 14,
    textAlign: 'center',
  },
});