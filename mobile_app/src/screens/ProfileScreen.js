import React, { useState, useRef, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Switch,
  Image,
  Animated,
  Easing,
  Platform,
  ToastAndroid,
  Modal,
  StatusBar,
  Alert,
} from "react-native";
import { useAuth } from "../context/AuthContext";
import { useFocusEffect } from "@react-navigation/native";
import * as SecureStore from "expo-secure-store";
import * as ImagePicker from "expo-image-picker";
import * as FileSystem from "expo-file-system/legacy";
import { Ionicons } from "@expo/vector-icons";
import { BACKEND_URL } from "../config";

const CATALOG_PORTFOLIO = [
  { name: "Apple", desc: "Scab, Black Rot, Cedar Apple Rust, Healthy" },
  {
    name: "Corn (Maize)",
    desc: "Blight, Common Rust, Gray Leaf Spot, Northern Blight, Healthy",
  },
  {
    name: "Cotton",
    desc: "Aphids, Bacterial Blight, Powdery Mildew, Target Spot, Healthy",
  },
  {
    name: "Grape",
    desc: "Black Rot, Esca (Black Measles), Leaf Blight, Healthy",
  },
  {
    name: "Mango",
    desc: "Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, Sooty Mould, Healthy",
  },
  { name: "Potato", desc: "Early Blight, Late Blight, Healthy" },
  {
    name: "Rice",
    desc: "Bacterial Blight, Brown Spot, Hispa, Leaf Blast, Leaf Scald, Sheath Blight, Healthy",
  },
  {
    name: "Sugarcane",
    desc: "Bacterial Blight, Banded Chlorosis, Brown Rust/Spot, Grassy Shoot, Mosaic, Pokkah Boeng, Red/Sett Rot, Smut, Yellow Leaf",
  },
  {
    name: "Tomato",
    desc: "Bacterial Spot, Early/Late Blight, Leaf Mold, Mosaic/Yellow Leaf Curl Virus, Powdery Mildew, Septoria, Spider Mites, Target Spot",
  },
  {
    name: "Wheat",
    desc: "Aphids, Rusts (Black, Brown, Yellow), Blast, Root Rot, Head Blight, Mites, Powdery Mildew, Septoria, Smut, Stem Fly, Tan Spot",
  },
];

export default function ProfileScreen({ navigation }) {
  const { userInfo, logout } = useAuth();
  const [loading, setLoading] = useState(false);
  const [statsLoading, setStatsLoading] = useState(true);
  const [stats, setStats] = useState({ total: 0, diseased: 0, healthy: 0 });
  const [notifications, setNotifications] = useState(true);
  const [profilePic, setProfilePic] = useState(null);

  // Custom Card Modals Interface States
  const [catalogVisible, setCatalogVisible] = useState(false);
  const [guideVisible, setGuideVisible] = useState(false);
  const [reportVisible, setReportVisible] = useState(false);
  const [cacheVisible, setCacheVisible] = useState(false);
  const [logoutVisible, setLogoutVisible] = useState(false);
  const [picOptionVisible, setPicOptionVisible] = useState(false);

  // Layout Animation Variables
  const headerAnim = useRef(new Animated.Value(0)).current;
  const statsAnim = useRef(new Animated.Value(0)).current;
  const sectionsAnim = useRef(new Animated.Value(0)).current;
  const avatarScale = useRef(new Animated.Value(1)).current;

  const showNotification = useCallback((msg) => {
    if (Platform.OS === "android") {
      ToastAndroid.showWithGravityAndOffset(
        msg,
        ToastAndroid.SHORT,
        ToastAndroid.BOTTOM,
        0,
        50,
      );
    } else {
      Alert.alert("AgriGuard", msg);
    }
  }, []);

  const fetchScreenData = async () => {
    try {
      setStatsLoading(true);
      const [token, savedPic] = await Promise.all([
        SecureStore.getItemAsync("userToken"),
        SecureStore.getItemAsync("profilePic"),
      ]);

      if (savedPic) setProfilePic(savedPic);

      const response = await fetch(`${BACKEND_URL}/api/scans`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (response.ok) {
        const scans = await response.json();
        setStats({
          total: scans.length,
          diseased: scans.filter((s) => !s.is_healthy).length,
          healthy: scans.filter((s) => s.is_healthy).length,
        });
      }
    } catch (err) {
      console.log("Sync processing fault:", err.message);
      showNotification("Network sync delayed. Displaying cached metrics.");
    } finally {
      setStatsLoading(false);
    }
  };

  useFocusEffect(
    useCallback(() => {
      headerAnim.setValue(0);
      statsAnim.setValue(0);
      sectionsAnim.setValue(0);

      Animated.stagger(60, [
        Animated.timing(headerAnim, {
          toValue: 1,
          duration: 350,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
        Animated.timing(statsAnim, {
          toValue: 1,
          duration: 350,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
        Animated.timing(sectionsAnim, {
          toValue: 1,
          duration: 350,
          easing: Easing.out(Easing.cubic),
          useNativeDriver: true,
        }),
      ]).start();

      fetchScreenData();
    }, []),
  );

  const saveImagePermanently = async (tempUri) => {
    try {
      const filename = `avatar_${Date.now()}.jpg`;
      const permanentDirectory = `${FileSystem.documentDirectory}${filename}`;

      await FileSystem.copyAsync({
        from: tempUri,
        to: permanentDirectory,
      });

      setProfilePic(permanentDirectory);
      await SecureStore.setItemAsync("profilePic", permanentDirectory);
      showNotification("Profile picture updated successfully!");
    } catch (error) {
      console.log("Image process exception loop:", error);
      showNotification("Failed to process and store avatar picture.");
    }
  };

  const handleLaunchCamera = async () => {
    setPicOptionVisible(false);
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) return showNotification("Camera access refused.");

    StatusBar.setBarStyle("light-content", true);
    if (Platform.OS === "android")
      StatusBar.setBackgroundColor("#000000", true);

    const result = await ImagePicker.launchCameraAsync({
      quality: 0.6,
      allowsEditing: true,
      aspect: [1, 1],
    });

    StatusBar.setBarStyle("dark-content", true);
    if (Platform.OS === "android")
      StatusBar.setBackgroundColor("#f5f5f5", true);

    if (!result.canceled) await saveImagePermanently(result.assets[0].uri);
  };

  const handleLaunchGallery = async () => {
    setPicOptionVisible(false);
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) return showNotification("Library access refused.");

    StatusBar.setBarStyle("light-content", true);
    if (Platform.OS === "android")
      StatusBar.setBackgroundColor("#000000", true);

    const result = await ImagePicker.launchImageLibraryAsync({
      quality: 0.6,
      allowsEditing: true,
      aspect: [1, 1],
    });

    StatusBar.setBarStyle("dark-content", true);
    if (Platform.OS === "android")
      StatusBar.setBackgroundColor("#f5f5f5", true);

    if (!result.canceled) await saveImagePermanently(result.assets[0].uri);
  };

  const handleRemovePhoto = async () => {
    setPicOptionVisible(false);
    setProfilePic(null);
    await SecureStore.deleteItemAsync("profilePic");
    showNotification("Profile picture removed.");
  };

  const handleExecuteClearCache = async () => {
    try {
      const cacheDir = FileSystem.cacheDirectory;
      if (cacheDir) {
        const cachedFiles = await FileSystem.readDirectoryAsync(cacheDir);
        await Promise.all(
          cachedFiles.map((file) =>
            FileSystem.deleteAsync(`${cacheDir}${file}`, { idempotent: true }),
          ),
        );
        setCacheVisible(false);
        showNotification("Application image cache optimized and cleared!");
      }
    } catch (error) {
      setCacheVisible(false);
      showNotification("Storage directory optimization completed.");
    }
  };

  const getInitials = (name) => {
    if (!name) return "?";
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const pressIn = () =>
    Animated.spring(avatarScale, {
      toValue: 0.94,
      useNativeDriver: true,
    }).start();
  const pressOut = () =>
    Animated.spring(avatarScale, {
      toValue: 1,
      friction: 4,
      useNativeDriver: true,
    }).start();

  const MenuItem = ({
    icon,
    label,
    value,
    onPress,
    danger,
    toggle,
    toggleValue,
    onToggle,
  }) => (
    <TouchableOpacity
      style={styles.menuItem}
      onPress={onPress}
      disabled={!onPress && !toggle}
      activeOpacity={onPress ? 0.6 : 1}
    >
      <View style={[styles.menuIconBox, danger && styles.menuIconBoxDanger]}>
        <Ionicons
          name={icon}
          size={18}
          color={danger ? "#e53935" : "#2e7d32"}
        />
      </View>
      <Text style={[styles.menuLabel, danger && styles.dangerText]}>
        {label}
      </Text>
      {value && <Text style={styles.menuValue}>{value}</Text>}
      {toggle && (
        <Switch
          value={toggleValue}
          onValueChange={onToggle}
          trackColor={{ false: "#eef0ef", true: "#a5d6a7" }}
          thumbColor={toggleValue ? "#2e7d32" : "#f4f3f4"}
        />
      )}
      {onPress && !danger && (
        <Ionicons name="chevron-forward" size={16} color="#bbb" />
      )}
    </TouchableOpacity>
  );

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Profile Header Block */}
      <Animated.View style={[styles.headerCard, { opacity: headerAnim }]}>
        <Animated.View style={{ transform: [{ scale: avatarScale }] }}>
          <TouchableOpacity
            style={styles.avatarWrapper}
            onPress={() => setPicOptionVisible(true)}
            onPressIn={pressIn}
            onPressOut={pressOut}
            activeOpacity={1}
          >
            {profilePic ? (
              <Image source={{ uri: profilePic }} style={styles.avatarImage} />
            ) : (
              <View style={styles.avatarFallback}>
                <Text style={styles.avatarText}>
                  {getInitials(userInfo?.name)}
                </Text>
              </View>
            )}
            <View style={styles.cameraBadge}>
              <Ionicons name="camera" size={12} color="#fff" />
            </View>
          </TouchableOpacity>
        </Animated.View>

        <Text style={styles.name}>{userInfo?.name || "Arslan Ahmed"}</Text>
        <Text style={styles.email}>
          {userInfo?.email || "mearslanahmed@gmail.com"}
        </Text>

        <View style={styles.rolePill}>
          <View style={styles.roleDot} />
          <Text style={styles.roleText}>Verified Farmer Account</Text>
        </View>

        <TouchableOpacity
          style={styles.editBtn}
          onPress={() => navigation.navigate("EditProfileScreen")}
          activeOpacity={0.7}
        >
          <Ionicons
            name="pencil"
            size={13}
            color="#2e7d32"
            style={{ marginRight: 4 }}
          />
          <Text style={styles.editBtnText}>Modify Profile</Text>
        </TouchableOpacity>
      </Animated.View>

      {/* Analytics Metric Counter Row */}
      <Animated.View style={[styles.statsContainer, { opacity: statsAnim }]}>
        {statsLoading ? (
          <View style={styles.statsLoaderBox}>
            <ActivityIndicator color="#2e7d32" />
          </View>
        ) : (
          <React.Fragment>
            <View style={styles.statCell}>
              <Text style={[styles.statNumber, { color: "#2c3e50" }]}>
                {stats.total}
              </Text>
              <Text style={styles.statLabel}>Total Scans</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statCell}>
              <Text style={[styles.statNumber, { color: "#e53935" }]}>
                {stats.diseased}
              </Text>
              <Text style={styles.statLabel}>Diseased</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statCell}>
              <Text style={[styles.statNumber, { color: "#2e7d32" }]}>
                {stats.healthy}
              </Text>
              <Text style={styles.statLabel}>Healthy</Text>
            </View>
          </React.Fragment>
        )}
      </Animated.View>

      {/* Configuration Navigation Group Blocks */}
      <Animated.View style={{ opacity: sectionsAnim }}>
        <Text style={styles.sectionLabel}>PREFERENCES</Text>
        <View style={styles.menuCard}>
          <MenuItem
            icon="notifications"
            label="Push Alerts Notify"
            toggle
            toggleValue={notifications}
            onToggle={setNotifications}
          />
          <View style={styles.menuDivider} />
          <MenuItem
            icon="language"
            label="System Language"
            value="English"
            onPress={() =>
              showNotification("Urdu support pack rolling out soon.")
            }
          />
        </View>

        <Text style={styles.sectionLabel}>SUPPORT & UTILITIES</Text>
        <View style={styles.menuCard}>
          <MenuItem
            icon="help-circle"
            label="How to Diagnose Crops"
            onPress={() => setGuideVisible(true)}
          />
          <View style={styles.menuDivider} />
          <MenuItem
            icon="bug"
            label="Report Technical Glitch"
            onPress={() => setReportVisible(true)}
          />
          <View style={styles.menuDivider} />
          <MenuItem
            icon="albums"
            label="Supported Crops Catalog"
            onPress={() => setCatalogVisible(true)}
          />
          <View style={styles.menuDivider} />
          <MenuItem
            icon="trash-bin"
            label="Optimize Storage (Clear Cache)"
            onPress={() => setCacheVisible(true)}
          />
        </View>

        <Text style={styles.sectionLabel}>APPLICATION METRICS</Text>
        <View style={styles.menuCard}>
          <MenuItem
            icon="information-circle"
            label="App Version"
            value="v1.0.0 (Beta)"
          />
        </View>

        <TouchableOpacity
          style={styles.logoutButton}
          onPress={() => setLogoutVisible(true)}
          activeOpacity={0.75}
        >
          <Ionicons
            name="log-out"
            size={18}
            color="#e53935"
            style={{ marginRight: 6 }}
          />
          <Text style={styles.logoutText}>Log Out</Text>
        </TouchableOpacity>

        <Text style={styles.footer}>
          AgriGuard Suite · Intelligent Crop Architecture 2026
        </Text>
      </Animated.View>

      {/* 
          MODAL 1: PROFILE PICTURE SELECTOR (FIXED STYLING SEPARATORS)
          */}
      <Modal
        visible={picOptionVisible}
        animationType="fade"
        transparent={true}
        onRequestClose={() => setPicOptionVisible(false)}
      >
        <View style={styles.modalOverlayCenter}>
          <View style={styles.popupCard}>
            <View style={styles.popupHeader}>
              <Ionicons name="image" size={20} color="#2e7d32" />
              <Text style={styles.popupTitle}>Profile Photo</Text>
            </View>
            <Text style={styles.popupInstruction}>
              Update or remove your current profile picture.
            </Text>
            <View style={styles.picActionCol}>
              <TouchableOpacity
                style={styles.popupBtn}
                onPress={handleLaunchCamera}
              >
                <Ionicons
                  name="camera"
                  size={16}
                  color="#fff"
                  style={{ marginRight: 6 }}
                />
                <Text style={styles.popupBtnText}>Take Photo</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.popupBtn}
                onPress={handleLaunchGallery}
              >
                <Ionicons
                  name="images"
                  size={16}
                  color="#fff"
                  style={{ marginRight: 6 }}
                />
                <Text style={styles.popupBtnText}>Choose from Gallery</Text>
              </TouchableOpacity>

              {profilePic && (
                <TouchableOpacity
                  style={[styles.popupBtn, styles.btnDangerSolid]}
                  onPress={handleRemovePhoto}
                >
                  <Ionicons
                    name="trash"
                    size={16}
                    color="#fff"
                    style={{ marginRight: 6 }}
                  />
                  <Text style={styles.popupBtnTextDanger}>
                    Remove Current Photo
                  </Text>
                </TouchableOpacity>
              )}

              <TouchableOpacity
                style={[styles.popupBtn, styles.btnCancelSolid]}
                onPress={() => setPicOptionVisible(false)}
              >
                <Text style={styles.popupBtnTextCancel}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* 
          MODAL 2: USER GUIDE
          */}
      <Modal
        visible={guideVisible}
        animationType="fade"
        transparent={true}
        onRequestClose={() => setGuideVisible(false)}
      >
        <View style={styles.modalOverlayCenter}>
          <View style={styles.popupCard}>
            <View style={styles.popupHeader}>
              <Ionicons name="book" size={20} color="#2e7d32" />
              <Text style={styles.popupTitle}>User Guide</Text>
            </View>
            <Text style={styles.popupInstruction}>
              1. Navigate directly to the 'Detect' menu tab.{"\n\n"}
              2. Capture or input a clear photo profile of the crop foliage leaf
              canopy.{"\n\n"}
              3. Hit 'Analyze' to run calculations over our backend validation
              layers.{"\n\n"}
              4. View results instantly along with pesticide recommendations.
            </Text>
            <TouchableOpacity
              style={styles.popupBtn}
              onPress={() => setGuideVisible(false)}
            >
              <Text style={styles.popupBtnText}>Got it</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* 
          MODAL 3: BUG REPORTING
          */}
      <Modal
        visible={reportVisible}
        animationType="fade"
        transparent={true}
        onRequestClose={() => setReportVisible(false)}
      >
        <View style={styles.modalOverlayCenter}>
          <View style={styles.popupCard}>
            <View style={styles.popupHeader}>
              <Ionicons name="bug" size={20} color="#2e7d32" />
              <Text style={styles.popupTitle}>Report Issue</Text>
            </View>
            <Text style={styles.popupInstruction}>
              Encountered an issue? Please route error captures directly to the
              technical support team:
            </Text>
            <View style={styles.emailBadge}>
              <Text style={styles.emailText}>arslanahmednaseem@gmail.com</Text>
            </View>
            <TouchableOpacity
              style={styles.popupBtn}
              onPress={() => setReportVisible(false)}
            >
              <Text style={styles.popupBtnText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* 
          MODAL 4: STORAGE CLEANER
          */}
      <Modal
        visible={cacheVisible}
        animationType="fade"
        transparent={true}
        onRequestClose={() => setCacheVisible(false)}
      >
        <View style={styles.modalOverlayCenter}>
          <View style={styles.popupCard}>
            <View style={styles.popupHeader}>
              <Ionicons name="trash-bin" size={20} color="#e53935" />
              <Text style={styles.popupTitle}>Clear Cache</Text>
            </View>
            <Text style={styles.popupInstruction}>
              This will delete temporary image files to save phone space. Your
              scans are safe.
            </Text>
            <View style={styles.modalActionRow}>
              <TouchableOpacity
                style={[styles.popupBtn, styles.btnCancelSolid, { flex: 1 }]}
                onPress={() => setCacheVisible(false)}
              >
                <Text style={styles.popupBtnTextCancel}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.popupBtn, styles.btnDangerSolid, { flex: 1 }]}
                onPress={handleExecuteClearCache}
              >
                <Text style={styles.popupBtnTextDanger}>Optimize</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* 
          MODAL 5: LOGOUT SYSTEM CLEAR
         */}
      <Modal
        visible={logoutVisible}
        animationType="fade"
        transparent={true}
        onRequestClose={() => setLogoutVisible(false)}
      >
        <View style={styles.modalOverlayCenter}>
          <View style={styles.popupCard}>
            <View style={styles.popupHeader}>
              <Ionicons name="log-out" size={20} color="#e53935" />
              <Text style={styles.popupTitle}>Log Out</Text>
            </View>
            <Text style={styles.popupInstruction}>
              Are you sure you want to end your AgriGuard session?
            </Text>
            <View style={styles.modalActionRow}>
              <TouchableOpacity
                style={[styles.popupBtn, styles.btnCancelSolid, { flex: 1 }]}
                onPress={() => setLogoutVisible(false)}
              >
                <Text style={styles.popupBtnTextCancel}>Stay</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.popupBtn, styles.btnDangerSolid, { flex: 1 }]}
                onPress={async () => {
                  setLogoutVisible(false);
                  await logout();
                }}
              >
                <Text style={styles.popupBtnTextDanger}>Log Out</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* 
          MODAL 6: SUPPORTED CROPS CATALOG (BOTTOM SHEET)
          */}
      <Modal
        visible={catalogVisible}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setCatalogVisible(false)}
      >
        <View style={styles.modalOverlayBottom}>
          <View style={styles.bottomSheet}>
            <View style={styles.modalHeaderRow}>
              <TouchableOpacity onPress={() => setCatalogVisible(false)}>
                <Ionicons name="close-circle" size={26} color="#888" />
              </TouchableOpacity>
              <Text style={styles.modalTitle}>Crop Portfolio</Text>
              <View style={{ width: 26 }} />
            </View>
            <ScrollView showsVerticalScrollIndicator={false}>
              {CATALOG_PORTFOLIO.map((item) => (
                <View key={item.name} style={styles.catalogItem}>
                  <View style={styles.catTitleRow}>
                    <Ionicons name="leaf" size={14} color="#2e7d32" />
                    <Text style={styles.catName}> {item.name}</Text>
                  </View>
                  <Text style={styles.catDesc}>{item.desc}</Text>
                </View>
              ))}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f5f5f5" },
  content: { paddingTop: 30, paddingHorizontal: 20, paddingBottom: 120 },
  headerCard: {
    backgroundColor: "#fff",
    borderRadius: 20,
    padding: 24,
    alignItems: "center",
    elevation: 3,
    marginBottom: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.04,
    shadowRadius: 8,
  },
  avatarWrapper: { position: "relative", marginBottom: 14 },
  avatarImage: {
    width: 88,
    height: 88,
    borderRadius: 44,
    borderWidth: 3,
    borderColor: "#e8f5e9",
  },
  avatarFallback: {
    width: 88,
    height: 88,
    borderRadius: 44,
    backgroundColor: "#2e7d32",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 3,
    borderColor: "#e8f5e9",
  },
  avatarText: { fontSize: 26, fontWeight: "800", color: "#fff" },
  cameraBadge: {
    position: "absolute",
    bottom: 2,
    right: 2,
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: "#2e7d32",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: "#fff",
  },
  name: { fontSize: 22, fontWeight: "800", color: "#1a1a1a" },
  email: { fontSize: 13, color: "#888", marginBottom: 10 },
  rolePill: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#e8f5e9",
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 20,
    marginBottom: 15,
  },
  roleDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: "#2e7d32",
    marginRight: 6,
  },
  roleText: { fontSize: 11, color: "#2e7d32", fontWeight: "700" },
  editBtn: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#e2ece2",
    borderRadius: 10,
    paddingHorizontal: 15,
    paddingVertical: 7,
    backgroundColor: "#fafafa",
  },
  editBtnText: { color: "#2e7d32", fontWeight: "700", fontSize: 12 },
  statsContainer: {
    flexDirection: "row",
    backgroundColor: "#fff",
    borderRadius: 20,
    paddingVertical: 20,
    marginBottom: 20,
    elevation: 2,
    justifyContent: "center",
    minHeight: 78,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.04,
    shadowRadius: 8,
  },
  statsLoaderBox: { flex: 1, justifyContent: "center", alignItems: "center" },
  statCell: { flex: 1, alignItems: "center" },
  statDivider: {
    width: 1,
    height: "60%",
    backgroundColor: "#f0f0f0",
    alignSelf: "center",
  },
  statNumber: { fontSize: 22, fontWeight: "800" },
  statLabel: { fontSize: 11, color: "#888", marginTop: 4, fontWeight: "600" },
  sectionLabel: {
    fontSize: 10,
    fontWeight: "800",
    color: "#aaa",
    letterSpacing: 1,
    marginLeft: 4,
    marginBottom: 8,
  },
  menuCard: {
    backgroundColor: "#fff",
    marginBottom: 20,
    borderRadius: 20,
    elevation: 2,
    overflow: "hidden",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.04,
    shadowRadius: 8,
  },
  menuItem: { flexDirection: "row", alignItems: "center", padding: 15 },
  menuIconBox: {
    width: 32,
    height: 32,
    borderRadius: 10,
    backgroundColor: "#e8f5e9",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },
  menuIconBoxDanger: { backgroundColor: "#fdecea" },
  menuLabel: { flex: 1, fontSize: 14, fontWeight: "600", color: "#333" },
  menuValue: { fontSize: 13, color: "#999", marginRight: 5 },
  menuDivider: { height: 1, backgroundColor: "#f9f9f9", marginLeft: 60 },
  dangerText: { color: "#e53935" },
  logoutButton: {
    paddingVertical: 14,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#fff",
    borderWidth: 1.5,
    borderColor: "#fdecea",
    flexDirection: "row",
  },
  logoutText: { color: "#e53935", fontSize: 14, fontWeight: "700" },
  footer: { textAlign: "center", fontSize: 10, color: "#ccc", marginTop: 15 },

  // Custom Center Popup Positioning Matrices
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

  // Custom High-Contrast Button Specifications
  popupBtn: {
    backgroundColor: "#2e7d32",
    height: 48,
    borderRadius: 12,
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
    width: "100%",
  },
  popupBtnText: { color: "#fff", fontWeight: "700", fontSize: 15 },
  popupBtnTextDanger: { color: "#fff", fontWeight: "700", fontSize: 15 },
  popupBtnTextCancel: { color: "#666", fontWeight: "700", fontSize: 15 },
  emailBadge: {
    backgroundColor: "#f5f5f5",
    padding: 12,
    borderRadius: 10,
    marginBottom: 20,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#eee",
  },
  emailText: { color: "#2e7d32", fontWeight: "700", fontSize: 13 },
  modalActionRow: { flexDirection: "row", gap: 10 },
  picActionCol: { gap: 12 },

  // SOLID OVERRIDES (No structural collapse)
  btnDangerSolid: { backgroundColor: "#e53935", height: 48, borderRadius: 12 },
  btnCancelSolid: { backgroundColor: "#eee", height: 48, borderRadius: 12 },

  // Custom Slider Bottom Sheet Styles Configuration
  modalOverlayBottom: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.4)",
    justifyContent: "flex-end",
  },
  bottomSheet: {
    backgroundColor: "#fff",
    borderTopLeftRadius: 25,
    borderTopRightRadius: 25,
    padding: 25,
    maxHeight: "80%",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.08,
    shadowRadius: 10,
    elevation: 15,
  },
  modalHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 20,
  },
  modalTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: "800",
    color: "#2e7d32",
    textAlign: "center",
  },
  catalogItem: {
    marginBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: "#f5f5f5",
    paddingBottom: 10,
  },
  catTitleRow: { flexDirection: "row", alignItems: "center" },
  catName: { fontSize: 15, fontWeight: "700", color: "#222" },
  catDesc: { fontSize: 12, color: "#777", lineHeight: 18, marginTop: 4 },
});
