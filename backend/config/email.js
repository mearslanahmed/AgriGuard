const sendEmail = async ({ to, subject, html }) => {
  const response = await fetch('https://api.brevo.com/v3/smtp/email', {
    method: 'POST',
    headers: {
      'api-key': process.env.BREVO_API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      sender: { email: process.env.VERIFIED_SENDER_EMAIL, name: 'AgriGuard Security' },
      to: [{ email: to }],
      subject,
      htmlContent: html
    })
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Brevo dispatch failed: ${err}`);
  }
};

module.exports = sendEmail;