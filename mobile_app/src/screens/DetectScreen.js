import React, { useState, useEffect, useRef } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  Image, ActivityIndicator, Alert, Animated,
  Dimensions, StatusBar, Easing
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { detectDisease } from '../services/detectService';

const { width } = Dimensions.get('window');
const IMAGE_BOX_HEIGHT = width * 0.85;

export default function DetectScreen({ navigation, onScanComplete }) {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  // Entrance animations
  const titleAnim = useRef(new Animated.Value(0)).current;
  const boxAnim = useRef(new Animated.Value(0)).current;
  const buttonsAnim = useRef(new Animated.Value(0)).current;

  // Scan line animation (runs when image is selected)
  const scanLine = useRef(new Animated.Value(0)).current;
  const scanLoop = useRef(null);

  // Placeholder pulse
  const pulse = useRef(new Animated.Value(1)).current;
  const pulseLoop = useRef(null);

  // Button press scales
  const cameraScale = useRef(new Animated.Value(1)).current;
  const galleryScale = useRef(new Animated.Value(1)).current;
  const analyzeScale = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    // Staggered entrance
    Animated.stagger(100, [
      Animated.timing(titleAnim, {
        toValue: 1, duration: 400,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(boxAnim, {
        toValue: 1, duration: 450,
        easing: Easing.out(Easing.back(1.2)),
        useNativeDriver: true,
      }),
      Animated.timing(buttonsAnim, {
        toValue: 1, duration: 400,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
    ]).start();

    // Placeholder pulse loop
    pulseLoop.current = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 0.6, duration: 900, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 1, duration: 900, useNativeDriver: true }),
      ])
    );
    pulseLoop.current.start();

    return () => {
      pulseLoop.current?.stop();
      scanLoop.current?.stop();
    };
  }, []);

  useEffect(() => {
    if (image) {
      pulseLoop.current?.stop();
      scanLine.setValue(0);
      scanLoop.current = Animated.loop(
        Animated.timing(scanLine, {
          toValue: 1, duration: 2000,
          easing: Easing.inOut(Easing.quad),
          useNativeDriver: true,
        })
      );
      scanLoop.current.start();
    } else {
      scanLoop.current?.stop();
      scanLine.setValue(0);
      pulseLoop.current?.start();
    }
  }, [image]);

  const pressIn = (scale) =>
    Animated.spring(scale, { toValue: 0.94, useNativeDriver: true }).start();
  const pressOut = (scale) =>
    Animated.spring(scale, { toValue: 1, friction: 4, useNativeDriver: true }).start();

  const pickFromGallery = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission required', 'Please allow gallery access.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'], quality: 0.8,
    });
    if (!result.canceled) setImage(result.assets[0].uri);
  };

  const pickFromCamera = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission required', 'Please allow camera access.');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ quality: 0.8 });
    if (!result.canceled) setImage(result.assets[0].uri);
  };

  const handleDetect = async () => {
    if (!image) return;
    setLoading(true);
    try {
      const { mlResult, pesticideData } = await detectDisease(image);
      if (onScanComplete) onScanComplete();
      navigation.navigate('Result', { mlResult, pesticideData, imageUri: image });
    } catch (err) {
      Alert.alert('Detection Failed', err.message);
    } finally {
      setLoading(false);
    }
  };

  const scanLineTranslate = scanLine.interpolate({
    inputRange: [0, 1],
    outputRange: [-IMAGE_BOX_HEIGHT / 2, IMAGE_BOX_HEIGHT / 2],
  });

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#f7faf7" />

      {/* Header */}
      <Animated.View style={[styles.header, {
        opacity: titleAnim,
        transform: [{ translateY: titleAnim.interpolate({ inputRange: [0, 1], outputRange: [-12, 0] }) }]
      }]}>
        <Text style={styles.title}>Detect Disease</Text>
        <Text style={styles.subtitle}>Point at a crop leaf to analyze</Text>
      </Animated.View>

      {/* Image Box */}
      <Animated.View style={[styles.imageBoxWrapper, {
        opacity: boxAnim,
        transform: [{ scale: boxAnim.interpolate({ inputRange: [0, 1], outputRange: [0.92, 1] }) }]
      }]}>
        <View style={[styles.imageBox, image && styles.imageBoxFilled]}>
          {image ? (
            <>
              <Image source={{ uri: image }} style={styles.image} />
              {/* Scan line overlay */}
              {loading ? (
                <View style={styles.loadingOverlay}>
                  <ActivityIndicator size="large" color="#fff" />
                  <Text style={styles.loadingText}>Analyzing...</Text>
                </View>
              ) : (
                <Animated.View style={[
                  styles.scanLine,
                  { transform: [{ translateY: scanLineTranslate }] }
                ]} />
              )}
            </>
          ) : (
            // Empty state
            <View style={styles.emptyState}>
              <Animated.View style={{ opacity: pulse }}>
                <View style={styles.iconCircle}>
                  <Ionicons name="leaf-outline" size={36} color="#2e7d32" />
                </View>
              </Animated.View>
              <Text style={styles.emptyTitle}>No image selected</Text>
              <Text style={styles.emptyHint}>Use camera or gallery below</Text>
            </View>
          )}
        </View>

        {/* Clear button — sits at bottom-right of image box */}
        {image && !loading && (
          <TouchableOpacity style={styles.clearBadge} onPress={() => setImage(null)}>
            <Ionicons name="close" size={14} color="#fff" />
          </TouchableOpacity>
        )}
      </Animated.View>

      {/* Bottom controls */}
      <Animated.View style={[styles.controls, {
        opacity: buttonsAnim,
        transform: [{ translateY: buttonsAnim.interpolate({ inputRange: [0, 1], outputRange: [20, 0] }) }]
      }]}>
        {/* Camera / Gallery row */}
        <View style={styles.row}>
          <Animated.View style={[styles.flex, { transform: [{ scale: cameraScale }] }]}>
            <TouchableOpacity
              style={styles.secondaryButton}
              onPress={pickFromCamera}
              onPressIn={() => pressIn(cameraScale)}
              onPressOut={() => pressOut(cameraScale)}
              activeOpacity={1}
            >
              <Ionicons name="camera-outline" size={18} color="#2e7d32" style={styles.btnIcon} />
              <Text style={styles.secondaryButtonText}>Camera</Text>
            </TouchableOpacity>
          </Animated.View>

          <Animated.View style={[styles.flex, { transform: [{ scale: galleryScale }] }]}>
            <TouchableOpacity
              style={styles.secondaryButton}
              onPress={pickFromGallery}
              onPressIn={() => pressIn(galleryScale)}
              onPressOut={() => pressOut(galleryScale)}
              activeOpacity={1}
            >
              <Ionicons name="images-outline" size={18} color="#2e7d32" style={styles.btnIcon} />
              <Text style={styles.secondaryButtonText}>Gallery</Text>
            </TouchableOpacity>
          </Animated.View>
        </View>

        {/* Analyze button */}
        <Animated.View style={{ transform: [{ scale: analyzeScale }] }}>
          <TouchableOpacity
            style={[styles.analyzeButton, (!image || loading) && styles.analyzeButtonDisabled]}
            onPress={handleDetect}
            onPressIn={() => image && pressIn(analyzeScale)}
            onPressOut={() => pressOut(analyzeScale)}
            disabled={loading || !image}
            activeOpacity={1}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <>
                <Ionicons
                  name="scan-outline"
                  size={18}
                  color={image ? '#fff' : '#a5d6a7'}
                  style={styles.btnIcon}
                />
                <Text style={[styles.analyzeButtonText, !image && styles.analyzeButtonTextDisabled]}>
                  Analyze Crop
                </Text>
              </>
            )}
          </TouchableOpacity>
        </Animated.View>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f7faf7',
    paddingHorizontal: 20,
    paddingTop: 30,
    paddingBottom: 100, // nav bar clearance
  },
  header: {
    marginBottom: 16,
  },
  title: {
    fontSize: 26,
    fontWeight: '700',
    color: '#1b5e20',
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
    marginTop: 2,
  },

  // Image box
  imageBoxWrapper: {
    position: 'relative',
    marginBottom: 16,
  },
  imageBox: {
    width: '100%',
    height: IMAGE_BOX_HEIGHT,
    borderRadius: 18,
    borderWidth: 1.5,
    borderColor: '#c8e6c9',
    borderStyle: 'dashed',
    backgroundColor: '#f0f9f0',
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
  },
  imageBoxFilled: {
    borderStyle: 'solid',
    borderColor: '#2e7d32',
    borderWidth: 2,
  },
  image: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },

  // Scan line
  scanLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 2,
    backgroundColor: 'rgba(76, 175, 80, 0.7)',
    shadowColor: '#4caf50',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 6,
  },

  // Loading overlay
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.45)',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 10,
  },
  loadingText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },

  // Clear badge
  clearBadge: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Empty state
  emptyState: {
    alignItems: 'center',
    gap: 8,
  },
  iconCircle: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#e8f5e9',
    borderWidth: 1.5,
    borderColor: '#c8e6c9',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 4,
  },
  emptyTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#555',
  },
  emptyHint: {
    fontSize: 13,
    color: '#aaa',
  },

  // Controls
  controls: {
    gap: 12,
  },
  row: {
    flexDirection: 'row',
    gap: 12,
  },
  flex: {
    flex: 1,
  },
  secondaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1.5,
    borderColor: '#2e7d32',
    borderRadius: 12,
    paddingVertical: 13,
    backgroundColor: '#fff',
  },
  secondaryButtonText: {
    color: '#2e7d32',
    fontWeight: '600',
    fontSize: 15,
  },
  btnIcon: {
    marginRight: 6,
  },
  analyzeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#2e7d32',
    paddingVertical: 15,
    borderRadius: 12,
  },
  analyzeButtonDisabled: {
    backgroundColor: '#c8e6c9',
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  analyzeButtonTextDisabled: {
    color: '#81c784',
  },
});