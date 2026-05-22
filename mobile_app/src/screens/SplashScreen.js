import React from 'react';
import { View, StyleSheet, Image, StatusBar } from 'react-native';

export default function SplashScreen() {
  return (
    <View style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />
      {/* Renders your sharp branding visual centered during framework loading */}
      <Image 
        source={require('../assets/splash.png')} 
        style={styles.splashImage} 
        resizeMode="contain"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  splashImage: {
    width: '75%',
    height: '75%',
  },
});