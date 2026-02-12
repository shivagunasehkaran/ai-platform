"""Mock tool functions for the travel planning agent."""

import random
from datetime import datetime, timedelta

from agent.schemas import (
    FlightResult,
    HotelResult,
    WeatherResult,
    AttractionResult,
)


def get_flights(origin: str, destination: str, date: str) -> list[FlightResult]:
    """Search for flights between cities.
    
    Args:
        origin: Origin city (e.g., "Wellington")
        destination: Destination city (e.g., "Auckland")
        date: Travel date (YYYY-MM-DD)
        
    Returns:
        list[FlightResult]: List of available flights.
    """
    # Mock flight data for NZ domestic routes
    airlines = ["Air New Zealand", "Jetstar", "Sounds Air"]
    
    flights = []
    
    # Generate 3-4 flight options
    for i in range(random.randint(3, 4)):
        airline = random.choice(airlines)
        hour = 6 + i * 4  # Flights at 6am, 10am, 2pm, 6pm
        
        base_price = 89 if airline == "Jetstar" else 129
        price = base_price + random.randint(-20, 50)
        
        flights.append(FlightResult(
            airline=airline,
            flight_number=f"{airline[:2].upper()}{random.randint(100, 999)}",
            price=float(price),
            departure=f"{hour:02d}:00",
            arrival=f"{hour + 1:02d}:15",
            origin=origin,
            destination=destination,
        ))
    
    return sorted(flights, key=lambda f: f.price)


def get_hotels(city: str, checkin: str, checkout: str, max_price: float = 300) -> list[HotelResult]:
    """Search for hotels in a city.
    
    Args:
        city: City to search (e.g., "Auckland")
        checkin: Check-in date (YYYY-MM-DD)
        checkout: Check-out date (YYYY-MM-DD)
        max_price: Maximum price per night in NZD
        
    Returns:
        list[HotelResult]: List of available hotels.
    """
    # Mock Auckland hotels
    auckland_hotels = [
        {
            "name": "CityLife Auckland",
            "price_per_night": 159.0,
            "rating": 4.2,
            "address": "171 Queen Street, Auckland CBD",
            "amenities": ["WiFi", "Gym", "Restaurant", "Parking"],
        },
        {
            "name": "Ibis Budget Auckland Central",
            "price_per_night": 89.0,
            "rating": 3.5,
            "address": "70 Beach Road, Auckland CBD",
            "amenities": ["WiFi", "24hr Reception"],
        },
        {
            "name": "YHA Auckland International",
            "price_per_night": 45.0,
            "rating": 4.0,
            "address": "5 Turner Street, Auckland CBD",
            "amenities": ["WiFi", "Kitchen", "Lounge", "Laundry"],
        },
        {
            "name": "Skycity Grand Hotel",
            "price_per_night": 289.0,
            "rating": 4.7,
            "address": "90 Federal Street, Auckland CBD",
            "amenities": ["WiFi", "Spa", "Pool", "Restaurant", "Casino"],
        },
        {
            "name": "Quest Auckland Serviced Apartments",
            "price_per_night": 175.0,
            "rating": 4.3,
            "address": "10 Commerce Street, Auckland CBD",
            "amenities": ["WiFi", "Kitchen", "Laundry", "Parking"],
        },
    ]
    
    # Filter by max price
    hotels = [
        HotelResult(**h) for h in auckland_hotels
        if h["price_per_night"] <= max_price
    ]
    
    return sorted(hotels, key=lambda h: h.price_per_night)


def get_weather(city: str, date: str) -> WeatherResult:
    """Get weather forecast for a city on a specific date.
    
    Args:
        city: City name (e.g., "Auckland")
        date: Date (YYYY-MM-DD)
        
    Returns:
        WeatherResult: Weather forecast.
    """
    # Mock Auckland weather (varies by season)
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Showers"]
    
    # Parse date to determine season
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        month = dt.month
    except ValueError:
        month = datetime.now().month
    
    # NZ seasons (southern hemisphere)
    if month in [12, 1, 2]:  # Summer
        condition = random.choice(["Sunny", "Partly Cloudy", "Sunny"])
        temp_high = random.uniform(22, 28)
        temp_low = random.uniform(15, 19)
        precip = random.randint(5, 25)
    elif month in [3, 4, 5]:  # Autumn
        condition = random.choice(["Partly Cloudy", "Cloudy", "Light Rain"])
        temp_high = random.uniform(18, 23)
        temp_low = random.uniform(12, 16)
        precip = random.randint(20, 45)
    elif month in [6, 7, 8]:  # Winter
        condition = random.choice(["Cloudy", "Light Rain", "Showers"])
        temp_high = random.uniform(12, 16)
        temp_low = random.uniform(7, 11)
        precip = random.randint(40, 70)
    else:  # Spring
        condition = random.choice(["Partly Cloudy", "Sunny", "Showers"])
        temp_high = random.uniform(16, 21)
        temp_low = random.uniform(10, 14)
        precip = random.randint(25, 50)
    
    return WeatherResult(
        date=date,
        condition=condition,
        temp_high=round(temp_high, 1),
        temp_low=round(temp_low, 1),
        precipitation_chance=precip,
    )


def get_attractions(city: str) -> list[AttractionResult]:
    """Get tourist attractions in a city.
    
    Args:
        city: City name (e.g., "Auckland")
        
    Returns:
        list[AttractionResult]: List of attractions.
    """
    # Mock Auckland attractions
    auckland_attractions = [
        AttractionResult(
            name="Sky Tower",
            category="landmark",
            price=32.0,
            rating=4.5,
            description="Iconic 328m tower with observation deck and stunning city views",
            duration_hours=1.5,
        ),
        AttractionResult(
            name="Auckland War Memorial Museum",
            category="museum",
            price=28.0,
            rating=4.7,
            description="World-class museum featuring Māori and Pacific Island collections",
            duration_hours=3.0,
        ),
        AttractionResult(
            name="Waiheke Island Ferry & Wine Tour",
            category="tour",
            price=85.0,
            rating=4.8,
            description="Scenic ferry ride and wine tasting at boutique vineyards",
            duration_hours=6.0,
        ),
        AttractionResult(
            name="Auckland Art Gallery",
            category="museum",
            price=0.0,
            rating=4.4,
            description="Free entry to NZ's largest art institution with over 15,000 works",
            duration_hours=2.0,
        ),
        AttractionResult(
            name="Mount Eden (Maungawhau)",
            category="nature",
            price=0.0,
            rating=4.6,
            description="Volcanic cone with panoramic views - free to visit",
            duration_hours=1.5,
        ),
        AttractionResult(
            name="Kelly Tarlton's Sea Life Aquarium",
            category="aquarium",
            price=44.0,
            rating=4.3,
            description="Underwater aquarium with sharks, penguins, and stingrays",
            duration_hours=2.5,
        ),
        AttractionResult(
            name="Viaduct Harbour",
            category="landmark",
            price=0.0,
            rating=4.2,
            description="Waterfront dining and entertainment precinct - free to explore",
            duration_hours=2.0,
        ),
        AttractionResult(
            name="One Tree Hill (Maungakiekie)",
            category="nature",
            price=0.0,
            rating=4.5,
            description="Historic volcanic cone and park with monument",
            duration_hours=2.0,
        ),
    ]
    
    return auckland_attractions


# Tool registry for the agent
TOOL_REGISTRY = {
    "get_flights": get_flights,
    "get_hotels": get_hotels,
    "get_weather": get_weather,
    "get_attractions": get_attractions,
}


# OpenAI function definitions
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_flights",
            "description": "Search for flights between two cities on a specific date",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Origin city (e.g., 'Wellington')",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination city (e.g., 'Auckland')",
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date in YYYY-MM-DD format",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hotels",
            "description": "Search for hotels in a city within a price range",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City to search for hotels",
                    },
                    "checkin": {
                        "type": "string",
                        "description": "Check-in date in YYYY-MM-DD format",
                    },
                    "checkout": {
                        "type": "string",
                        "description": "Check-out date in YYYY-MM-DD format",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price per night in NZD",
                    },
                },
                "required": ["city", "checkin", "checkout"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather forecast for a city on a specific date",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format",
                    },
                },
                "required": ["city", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attractions",
            "description": "Get list of tourist attractions in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    },
                },
                "required": ["city"],
            },
        },
    },
]
