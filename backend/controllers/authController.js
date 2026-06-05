const jwt = require("jsonwebtoken");
const bcrypt = require("bcryptjs");
const User = require("../models/User");
const OTP = require("../models/OTP");
const sendEmail = require("../config/email");

const generateToken = (userId) => {
  return jwt.sign({ id: userId }, process.env.JWT_SECRET, { expiresIn: "30d" });
};

// POST /api/auth/send-otp
const sendVerificationOTP = async (req, res) => {
  try {
    const { email, purpose } = req.body;

    if (!email || !purpose) {
      return res
        .status(400)
        .json({ message: "Email and transaction intent purpose required." });
    }

    const emailLower = email.toLowerCase().trim();
    const userExists = await User.findOne({ email: emailLower });

    if (purpose === "register" && userExists) {
      return res
        .status(400)
        .json({ message: "This email address is already registered." });
    }
    if (purpose === "reset" && !userExists) {
      return res
        .status(404)
        .json({ message: "No registered user profile found with this email." });
    }

    // Generate cryptographic 6-digit numeric token string
    const otpCode = Math.floor(100000 + Math.random() * 900000).toString();
    const hashedOtp = await bcrypt.hash(otpCode, 10);

    // Evict old unused tokens for this context to block spamming pipelines
    await OTP.deleteMany({ email: emailLower, purpose });
    await OTP.create({ email: emailLower, otp: hashedOtp, purpose });

    const mailOptions = {
      from: `"AgriGuard Security" <${process.env.EMAIL_USER}>`,
      to: emailLower,
      subject:
        purpose === "register"
          ? "Verify Your AgriGuard Account"
          : "Reset Your AgriGuard Password",
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto; border: 1px solid #eef0ef; padding: 24px; border-radius: 16px;">
          <h2 style="color: #2e7d32; text-align: center; margin-top: 0;">AgriGuard Security</h2>
          <p style="color: #333; font-size: 15px;">Hello,</p>
          <p style="color: #555; font-size: 14px; line-height: 20px;">Use the following unique verification code to authorize your action. This credential block expires in <strong>5 minutes</strong>.</p>
          <div style="background-color: #f4f6f4; padding: 16px; text-align: center; font-size: 32px; font-weight: 800; letter-spacing: 6px; color: #2e7d32; margin: 24px 0; border-radius: 12px; border: 1px solid #e2ece2;">
            ${otpCode}
          </div>
          <p style="font-size: 11px; color: #999; text-align: center; margin-bottom: 0;">If you did not issue this verification query, please discard this envelope securely.</p>
        </div>
      `,
    };

    await sendEmail({
      to: emailLower,
      subject: mailOptions.subject,
      html: mailOptions.html,
    });
    res.json({
      success: true,
      message: "Verification transaction key dispatched to your inbox.",
    });
  } catch (error) {
    console.error("OTP Dispatch Failure:", error.message);
    res
      .status(500)
      .json({ message: "Failed to deliver secure email token layer." });
  }
};

// POST /api/auth/register
const register = async (req, res) => {
  try {
    const { name, email, password, otp } = req.body;

    if (!name || !email || !password || !otp) {
      return res
        .status(400)
        .json({
          message: "All entry blocks including validation token required.",
        });
    }

    const emailLower = email.toLowerCase().trim();

    const record = await OTP.findOne({
      email: emailLower,
      purpose: "register",
    });
    if (!record) {
      return res
        .status(400)
        .json({
          message: "Token lifetime expired. Issue a fresh code request.",
        });
    }

    const isMatch = await bcrypt.compare(otp, record.otp);
    if (!isMatch)
      return res
        .status(401)
        .json({ message: "Invalid token verification signature." });

    const existingUser = await User.findOne({ email: emailLower });
    if (existingUser)
      return res.status(400).json({ message: "This email is already active." });

    const user = await User.create({ name, email: emailLower, password });
    await OTP.deleteOne({ _id: record._id });

    res.status(201).json({
      token: generateToken(user._id),
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
      },
    });
  } catch (error) {
    res.status(500).json({ message: "Server registration fault." });
  }
};

// POST /api/auth/reset-password
const resetPasswordWithOTP = async (req, res) => {
  try {
    const { email, otp, newPassword } = req.body;

    if (!email || !otp || !newPassword) {
      return res
        .status(400)
        .json({ message: "All adjustment values required." });
    }
    if (newPassword.length < 6) {
      return res
        .status(400)
        .json({
          message: "New password constraints specify 6 characters minimum.",
        });
    }

    const emailLower = email.toLowerCase().trim();

    const record = await OTP.findOne({ email: emailLower, purpose: "reset" });
    if (!record)
      return res
        .status(400)
        .json({ message: "Token expired or missing trace parameters." });

    const isMatch = await bcrypt.compare(otp, record.otp);
    if (!isMatch)
      return res
        .status(401)
        .json({ message: "Invalid token verification signature." });

    const user = await User.findOne({ email: emailLower });
    if (!user)
      return res.status(404).json({ message: "User reference not found." });

    user.password = newPassword;
    await user.save();
    await OTP.deleteOne({ _id: record._id });

    res.json({
      success: true,
      message: "Password reset validated. Proceed to login.",
    });
  } catch (error) {
    res.status(500).json({ message: "Credential modification crash." });
  }
};

const login = async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email: email.toLowerCase().trim() });
    if (!user)
      return res.status(401).json({ message: "Invalid email or password." });

    const isMatch = await user.matchPassword(password);
    if (!isMatch)
      return res.status(401).json({ message: "Invalid email or password." });

    res.json({
      token: generateToken(user._id),
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
      },
    });
  } catch (error) {
    res.status(500).json({ message: "Authentication failure." });
  }
};

const getMe = async (req, res) => {
  res.json({
    id: req.user._id,
    name: req.user.name,
    email: req.user.email,
    role: req.user.role,
  });
};

const updateProfile = async (req, res) => {
  try {
    const { name, email } = req.body;
    const user = await User.findById(req.user._id);
    if (name) user.name = name;
    if (email) user.email = email;
    await user.save();
    res.json({
      id: user._id,
      name: user.name,
      email: user.email,
      role: user.role,
    });
  } catch (error) {
    res.status(500).json({ message: "Profile save fail." });
  }
};

const changePassword = async (req, res) => {
  try {
    const { currentPassword, newPassword } = req.body;
    const user = await User.findById(req.user._id);
    const isMatch = await user.matchPassword(currentPassword);
    if (!isMatch)
      return res
        .status(401)
        .json({ message: "Current password match failed." });
    user.password = newPassword;
    await user.save();
    res.json({ message: "Password change tracking success." });
  } catch (error) {
    res
      .status(500)
      .json({ message: "Password modification sequence dropped." });
  }
};

module.exports = {
  sendVerificationOTP,
  register,
  login,
  getMe,
  updateProfile,
  changePassword,
  resetPasswordWithOTP,
};
