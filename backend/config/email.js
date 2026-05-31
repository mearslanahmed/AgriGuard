const nodemailer = require('nodemailer');

const transporter = nodemailer.createTransport({
  host: 'smtp-relay.brevo.com',
  port: 587,
  secure: false,
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_APP_PASS,
  },
});

transporter.verify((error, success) => {
  if (error) {
    console.error('Mail Carrier Relay Refused Handshake:', error.message);
  } else {
    console.log('AgriGuard Secure Mail Carrier Initialized and Active');
  }
});

module.exports = transporter;