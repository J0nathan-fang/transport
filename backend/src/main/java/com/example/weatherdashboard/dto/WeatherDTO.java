package com.example.weatherdashboard.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class WeatherDTO {
    private String locationName;
    private String temp;
    private String weatherText;
    private String feelsLike;
    private String vis; // visibility
    private String humidity;
    private String windSpeed;
    private String windDir;
}
