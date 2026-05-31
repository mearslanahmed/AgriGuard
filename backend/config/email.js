const axios = require('axios');

const sendEmailViaApi = async (email, subject, content) => {
  try {
    // This sends an HTTPS POST request, which Render does NOT block.
    await axios.post('https://api.brevo.com/v3/smtp/email', {
      sender: { 
        name: "AgriGuard", 
        email: process.env.VERIFIED_SENDER_EMAIL 
      },
      to: [{ email: email }],
      subject: subject,
      textContent: content
    }, {
      headers: { 
        'api-key': process.env.BREVO_API_KEY,
        'Content-Type': 'application/json' 
      }
    });
    return true;
  } catch (error) {
    // Detailed error logging for Render dashboard
    console.error('Brevo API Error:', error.response?.data || error.message);
    throw error;
  }
};

module.exports = { sendEmailViaApi };