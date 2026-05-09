package com.example.weatherdashboard.service;

import com.example.weatherdashboard.dto.QWeatherResponse;
import com.example.weatherdashboard.dto.WeatherDTO;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

@Service
public class WeatherService {

    @Value("${qweather.api-host}")
    private String apiHost;

    private final RestTemplate restTemplate = new RestTemplate();
    private final JwtService jwtService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Autowired
    public WeatherService(JwtService jwtService) {
        this.jwtService = jwtService;
    }

    public WeatherDTO getWeatherData(String lat, String lon) {
        try {
            String token = jwtService.generateJwt();
            String url = String.format("https://%s/v7/weather/now?location=%s,%s", apiHost, lon, lat);

            HttpHeaders headers = new HttpHeaders();
            headers.set("Authorization", "Bearer " + token);
            headers.set("Accept-Encoding", "gzip");
            HttpEntity<String> entity = new HttpEntity<>(headers);

            ResponseEntity<byte[]> response = restTemplate.exchange(url, HttpMethod.GET, entity, byte[].class);
            byte[] body = response.getBody();
            String jsonResponse;

            if (body != null && body.length > 2 && body[0] == (byte) 0x1F && body[1] == (byte) 0x8B) {
                try (GZIPInputStream gis = new GZIPInputStream(new ByteArrayInputStream(body));
                     BufferedReader br = new BufferedReader(new InputStreamReader(gis, StandardCharsets.UTF_8))) {
                    jsonResponse = br.lines().collect(Collectors.joining());
                }
            } else if (body != null) {
                jsonResponse = new String(body, StandardCharsets.UTF_8);
            } else {
                return null;
            }

            QWeatherResponse qWeatherResponse = objectMapper.readValue(jsonResponse, QWeatherResponse.class);

            if (qWeatherResponse != null && qWeatherResponse.getNow() != null) {
                QWeatherResponse.Now now = qWeatherResponse.getNow();
                // Map all available data from QWeather to our new flat DTO
                return new WeatherDTO(
                        "点击位置", // QWeather free API doesn't provide location name by coords
                        now.getTemp(),
                        now.getText(),
                        now.getFeelsLike(),
                        now.getVis(),
                        now.getHumidity(),
                        now.getWindSpeed(),
                        now.getWindDir()
                );
            }
            return null;
        } catch (Exception e) {
            throw new RuntimeException("Error fetching or processing weather data", e);
        }
    }
}
