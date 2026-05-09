package com.example.weatherdashboard.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class QWeatherResponse {
    private Now now;

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Now {
        private String temp;
        private String text;
        private String feelsLike;
        private String vis;
        private String humidity;
        private String windSpeed;
        private String windDir;
    }
}
