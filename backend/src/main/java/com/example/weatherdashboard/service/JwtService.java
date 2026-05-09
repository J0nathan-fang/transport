package com.example.weatherdashboard.service;

import io.jsonwebtoken.Jwts;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.security.KeyFactory;
import java.security.PrivateKey;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.Base64;
import java.util.Date;

@Service
public class JwtService {

    @Value("${qweather.project-id}")
    private String projectId;

    @Value("${qweather.key-id}")
    private String keyId;

    @Value("${qweather.private-key}")
    private String privateKeyString;

    public String generateJwt() throws Exception {
        long nowMillis = System.currentTimeMillis();
        Date now = new Date(nowMillis);
        // Per QWeather docs, expiry is 15-30 minutes. Let's use 15 minutes.
        long expMillis = nowMillis + 900_000;
        Date exp = new Date(expMillis);

        // 1. Clean the private key string from PEM format
        String pkcs8Pem = privateKeyString
                .replace("-----BEGIN PRIVATE KEY-----", "")
                .replace("-----END PRIVATE KEY-----", "")
                .replaceAll("\\s+", ""); // Use regex to remove all whitespace chars

        // 2. Decode the Base64 string to get the key bytes
        byte[] pkcs8EncodedBytes = Base64.getDecoder().decode(pkcs8Pem);

        // 3. Generate the PrivateKey object
        PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(pkcs8EncodedBytes);
        KeyFactory kf = KeyFactory.getInstance("Ed25519");
        PrivateKey privateKey = kf.generatePrivate(keySpec);

        // 4. Build the JWT
        return Jwts.builder()
                .header().keyId(keyId).and()
                .subject(projectId)
                .issuedAt(now)
                .expiration(exp)
                .signWith(privateKey)
                .compact();
    }
}
