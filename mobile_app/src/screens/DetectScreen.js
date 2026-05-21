import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Animated,
  Dimensions,
  StatusBar,
  Easing,
  Platform,
  Modal,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { Ionicons } from "@expo/vector-icons";
import { detectDisease } from "../services/detectService";

const { width } = Dimensions.get("window");
const IMAGE_BOX_HEIGHT = width * 0.85;

export default function DetectScreen({ navigation, onScanComplete }) {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  // Custom Modal States for Error Interception
  const [errorModalVisible, setErrorModalVisible] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

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
    Animated.stagger(100, [
      Animated.timing(titleAnim, {
        toValue: 1,
        duration: 400,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(boxAnim, {
        toValue: 1,
        duration: 450,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(buttonsAnim, {
        toValue: 1,
        duration: 400,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
    ]).start();

    pulseLoop.current = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, {
          toValue: 0.6,
          duration: 900,
          useNativeDriver: true,
        }),
        Animated.timing(pulse, {
          toValue: 1,
          duration: 900,
          useNativeDriver: true,
        }),
      ]),
    );
    pulseLoop.current.start();

    return () => {
      pulseLoop.current?.stop();
      scanLoop.current?.stop();
    };
  }, []);

  useEffect(() => {
    const unsubscribe = navigation.addListener("focus", () => {
      setImage(null);
    });
    return unsubscribe;
  }, [navigation]);

  useEffect(() => {
    if (image) {
      pulseLoop.current?.stop();
      scanLine.setValue(0);
      scanLoop.current = Animated.loop(
        Animated.timing(scanLine, {
          toValue: 1,
          duration: 2000,
          easing: Easing.inOut(Easing.quad),
          useNativeDriver: true,
        }),
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
    Animated.spring(scale, {
      toValue: 1,
      friction: 4,
      useNativeDriver: true,
    }).start();

  const pickFromGallery = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) return;
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 0.8,
    });
    if (!result.canceled) setImage(result.assets[0].uri);
  };

  const pickFromCamera = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) return;
    const result = await ImagePicker.launchCameraAsync({ quality: 0.8 });
    if (!result.canceled) setImage(result.assets[0].uri);
  };

  const handleDetect = async () => {
    if (!image) return;
    setLoading(true);
    try {
      const { mlResult, pesticideData } = await detectDisease(image);

      // FIXED: Only call execution hooks when detection succeeds to avoid unwanted logs
      if (onScanComplete) onScanComplete();

      navigation.navigate("Result", {
        mlResult,
        pesticideData,
        imageUri: image,
      });
    } catch (err) {
      setErrorMessage(err.message);
      setErrorModalVisible(true);
    } finally {
      setLoading(false);
    }
  };

  const scanLineTranslate = scanLine.interpolate({
    inputRange: [0, 1],
    outputRange: [-IMAGE_BOX_HEIGHT / 2, IMAGE_BOX_HEIGHT / 2],
  });

  const animStyle = (anim, slide = 12) => ({
    opacity: anim,
    transform: [
      {
        translateY: anim.interpolate({
          inputRange: [0, 1],
          outputRange: [slide, 0],
        }),
      },
    ],
  });

  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#fff" />

      {/* Flat Premium Header Layout */}
      <Animated.View style={[styles.header, animStyle(titleAnim, -10)]}>
        <Text style={styles.title}>Detect Disease</Text>
        <Text style={styles.subtitle}>Point at a crop leaf to analyze</Text>
      </Animated.View>

      <View style={styles.scrollContent}>
        {/* Modern Image Box Card */}
        <Animated.View style={[styles.imageBoxWrapper, animStyle(boxAnim, 16)]}>
          <View style={[styles.imageBox, image && styles.imageBoxFilled]}>
            {image ? (
              <>
                <Image source={{ uri: image }} style={styles.image} />
                {loading ? (
                  <View style={styles.loadingOverlay}>
                    <ActivityIndicator size="large" color="#fff" />
                    <Text style={styles.loadingText}>
                      Running AI Analysis...
                    </Text>
                  </View>
                ) : (
                  <Animated.View
                    style={[
                      styles.scanLine,
                      { transform: [{ translateY: scanLineTranslate }] },
                    ]}
                  />
                )}
              </>
            ) : (
              <View style={styles.emptyState}>
                <Animated.View style={{ opacity: pulse }}>
                  <View style={styles.iconCircle}>
                    <Ionicons name="leaf-outline" size={32} color="#2e7d32" />
                  </View>
                </Animated.View>
                <Text style={styles.emptyTitle}>No leaf image selected</Text>
                <Text style={styles.emptyHint}>
                  Take a photo or choose from your gallery below
                </Text>
              </View>
            )}
          </View>

          {image && !loading && (
            <TouchableOpacity
              style={styles.clearBadge}
              onPress={() => setImage(null)}
              activeOpacity={0.7}
            >
              <Ionicons name="close" size={14} color="#fff" />
            </TouchableOpacity>
          )}
        </Animated.View>

        {/* Action Controls */}
        <Animated.View style={[styles.controls, animStyle(buttonsAnim, 20)]}>
          <View style={styles.row}>
            <Animated.View
              style={[styles.flex, { transform: [{ scale: cameraScale }] }]}
            >
              <TouchableOpacity
                style={styles.secondaryButton}
                onPress={pickFromCamera}
                onPressIn={() => pressIn(cameraScale)}
                onPressOut={() => pressOut(cameraScale)}
                activeOpacity={1}
              >
                <Ionicons
                  name="camera"
                  size={18}
                  color="#2e7d32"
                  style={styles.btnIcon}
                />
                <Text style={styles.secondaryButtonText}>Camera</Text>
              </TouchableOpacity>
            </Animated.View>

            <Animated.View
              style={[styles.flex, { transform: [{ scale: galleryScale }] }]}
            >
              <TouchableOpacity
                style={styles.secondaryButton}
                onPress={pickFromGallery}
                onPressIn={() => pressIn(galleryScale)}
                onPressOut={() => pressOut(galleryScale)}
                activeOpacity={1}
              >
                <Ionicons
                  name="images"
                  size={18}
                  color="#2e7d32"
                  style={styles.btnIcon}
                />
                <Text style={styles.secondaryButtonText}>Gallery</Text>
              </TouchableOpacity>
            </Animated.View>
          </View>

          <Animated.View style={{ transform: [{ scale: analyzeScale }] }}>
            <TouchableOpacity
              style={[
                styles.analyzeButton,
                (!image || loading) && styles.analyzeButtonDisabled,
              ]}
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
                    name="scan"
                    size={18}
                    color={image ? "#fff" : "#a5d6a7"}
                    style={styles.btnIcon}
                  />
                  <Text
                    style={[
                      styles.analyzeButtonText,
                      !image && styles.analyzeButtonTextDisabled,
                    ]}
                  >
                    Analyze Crop Leaf
                  </Text>
                </>
              )}
            </TouchableOpacity>
          </Animated.View>
        </Animated.View>
      </View>

      {/* ========================================================
          BEAUTIFUL OVERLAY CARD MODAL RESTORED
         ======================================================== */}
      <Modal
        visible={errorModalVisible}
        animationType="fade"
        transparent={true}
        onRequestClose={() => setErrorModalVisible(false)}
      >
        <View style={styles.modalOverlayCenter}>
          <View style={styles.popupCard}>
            <View style={styles.popupHeader}>
              <Ionicons name="alert-circle" size={22} color="#e53935" />
              <Text style={styles.popupTitle}>Analysis Alert</Text>
            </View>
            <Text style={styles.popupInstruction}>{errorMessage}</Text>
            <TouchableOpacity
              style={[styles.popupBtn, styles.btnDangerSolid]}
              onPress={() => setErrorModalVisible(false)}
            >
              <Text style={styles.popupBtnTextDanger}>Try Again</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f5f5f5" },
  scrollContent: { paddingHorizontal: 20, paddingTop: 20 },
  header: {
    paddingHorizontal: 16,
    paddingTop:
      Platform.OS === "ios"
        ? 52
        : StatusBar.currentHeight
          ? StatusBar.currentHeight + 12
          : 16,
    paddingBottom: 14,
    backgroundColor: "#fff",
    borderBottomWidth: 1,
    borderBottomColor: "#eef0ef",
  },
  title: {
    fontSize: 16,
    fontWeight: "800",
    color: "#2e7d32",
    textAlign: "center",
  },
  subtitle: {
    fontSize: 12,
    color: "#888",
    marginTop: 2,
    fontWeight: "500",
    textAlign: "center",
  },
  imageBoxWrapper: { position: "relative", marginBottom: 20 },
  imageBox: {
    width: "100%",
    height: IMAGE_BOX_HEIGHT,
    borderRadius: 24,
    backgroundColor: "#fff",
    overflow: "hidden",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#eef0ef",
    elevation: 3,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.04,
    shadowRadius: 8,
  },
  imageBoxFilled: { borderColor: "#2e7d32", borderWidth: 1.5 },
  image: { width: "100%", height: "100%", resizeMode: "cover" },
  scanLine: {
    position: "absolute",
    left: 0,
    right: 0,
    height: 3,
    backgroundColor: "#2e7d32",
    opacity: 0.8,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0,0,0,0.6)",
    justifyContent: "center",
    alignItems: "center",
    gap: 12,
  },
  loadingText: { color: "#fff", fontSize: 15, fontWeight: "700" },
  clearBadge: {
    position: "absolute",
    top: 12,
    right: 12,
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "center",
    alignItems: "center",
  },
  emptyState: { alignItems: "center", paddingHorizontal: 20 },
  iconCircle: {
    width: 68,
    height: 68,
    borderRadius: 34,
    backgroundColor: "#e8f5e9",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 12,
  },
  emptyTitle: { fontSize: 15, fontWeight: "800", color: "#333" },
  emptyHint: {
    fontSize: 12,
    color: "#999",
    textAlign: "center",
    marginTop: 4,
    lineHeight: 18,
    fontWeight: "500",
  },
  controls: { gap: 12 },
  row: { flexDirection: "row", gap: 12 },
  flex: { flex: 1 },
  secondaryButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1.5,
    borderColor: "#e2ece2",
    borderRadius: 12,
    paddingVertical: 13,
    backgroundColor: "#fff",
    height: 50,
    elevation: 1,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.02,
    shadowRadius: 4,
  },
  secondaryButtonText: { color: "#2e7d32", fontWeight: "700", fontSize: 15 },
  btnIcon: { marginRight: 6 },
  analyzeButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#2e7d32",
    height: 50,
    borderRadius: 12,
  },
  analyzeButtonDisabled: { backgroundColor: "#c8e6c9" },
  analyzeButtonText: { color: "#fff", fontSize: 15, fontWeight: "700" },
  analyzeButtonTextDisabled: { color: "#81c784" },
  modalOverlayCenter: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  popupCard: {
    backgroundColor: "#fff",
    borderRadius: 24,
    padding: 24,
    width: "100%",
    maxWidth: 320,
    elevation: 10,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
  },
  popupHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: "#f5f5f5",
    paddingBottom: 10,
  },
  popupTitle: {
    fontSize: 18,
    fontWeight: "800",
    color: "#333",
    marginLeft: 10,
  },
  popupInstruction: {
    fontSize: 14,
    color: "#666",
    lineHeight: 22,
    marginBottom: 20,
  },
  popupBtn: {
    height: 48,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
  },
  btnDangerSolid: { backgroundColor: "#e53935" },
  popupBtnTextDanger: { color: "#fff", fontWeight: "700", fontSize: 15 },
});
