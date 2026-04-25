import React, { useState } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  Image, ActivityIndicator, Alert, ScrollView
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { detectDisease } from '../services/detectService';

export default function DetectScreen({ navigation }) {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const pickFromGallery = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission required', 'Please allow gallery access.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.8,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const pickFromCamera = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission required', 'Please allow camera access.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      quality: 0.8,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const handleDetect = async () => {
    if (!image){
      Alert.alert('No image', 'Please select or capture an image first');
      return;
    }

    setLoading(true);
    try {
      const [mlResult, pesticideData] = await detectDisease(image);
      // Pass result + image to result screen
      navigation.navigate('Result', { mlResult, pesticideData, imageUri: image });
    } catch (err) {
      Alert.alert('Detection Failed', err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Detect Disease</Text>
      <Text style={styles.subtitle}>Take or upload a crop leaf image</Text>

      {/* Image preview */}
      <View style={styles.imageBox}>
        {image ? (
          <Image source={{ uri: image }} style={styles.image} />
        ) : (
          <Text style={styles.placeholder}>No image selected</Text>
        )}
      </View>

      {/* Pick options */}
      <View style={styles.row}>
        <TouchableOpacity style={styles.secondaryButton} onPress={pickFromCamera}>
          <Text style={styles.secondaryButtonText}>Camera</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.secondaryButton} onPress={pickFromGallery}>
          <Text style={styles.secondaryButtonText}>Gallery</Text>
        </TouchableOpacity>
      </View>

      {/* Detect button */}
      <TouchableOpacity
        style={[styles.button, !image && styles.buttonDisabled]}
        onPress={handleDetect}
        disabled={loading || !image}
      >
        {loading
          ? <ActivityIndicator color="#fff" />
          : <Text style={styles.buttonText}>Analyze Crop</Text>
        }
      </TouchableOpacity>

      {/* Reset */}
      {image && !loading && (
        <TouchableOpacity onPress={() => setImage(null)}>
          <Text style={styles.resetText}>Clear image</Text>
        </TouchableOpacity>
      )}
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
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#2e7d32',
    marginTop: 20,
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 24,
  },
  imageBox: {
    width: '100%',
    height: 260,
    backgroundColor: '#e8f5e9',
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#c8e6c9',
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  placeholder: {
    color: '#aaa',
    fontSize: 15,
  },
  row: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  secondaryButton: {
    flex: 1,
    borderWidth: 2,
    borderColor: '#2e7d32',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#2e7d32',
    fontWeight: '600',
    fontSize: 15,
  },
  button: {
    backgroundColor: '#2e7d32',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
    width: '100%',
    marginBottom: 14,
  },
  buttonDisabled: {
    backgroundColor: '#a5d6a7',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  resetText: {
    color: '#e53935',
    fontSize: 14,
  },
});