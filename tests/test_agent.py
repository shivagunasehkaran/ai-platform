"""Tests for the planning agent."""

import json
import pytest
from pydantic import ValidationError

from agent.schemas import (
    Itinerary,
    ItineraryDay,
    FlightResult,
    HotelResult,
    WeatherResult,
    AttractionResult,
    TripConstraints,
)
from agent.tools import get_flights, get_hotels, get_weather, get_attractions
from agent.planner import parse_trip_request


class TestBudgetConstraint:
    """Tests for budget constraint handling."""

    def test_budget_parsing_nzd_format(self) -> None:
        """Verify budget is parsed from NZ$X format."""
        params = parse_trip_request("Plan a 2-day trip to Auckland for under NZ$500")
        assert params["budget"] == 500

    def test_budget_parsing_dollar_format(self) -> None:
        """Verify budget is parsed from $X format."""
        params = parse_trip_request("Plan a trip to Auckland for $750")
        assert params["budget"] == 750

    def test_budget_parsing_under_format(self) -> None:
        """Verify budget is parsed from 'under X' format."""
        params = parse_trip_request("Auckland trip under 600")
        assert params["budget"] == 600

    def test_budget_default_when_not_specified(self) -> None:
        """Verify default budget when not specified."""
        params = parse_trip_request("Plan a trip to Auckland")
        assert params["budget"] == 500  # Default value

    def test_days_parsing(self) -> None:
        """Verify days are parsed correctly."""
        params = parse_trip_request("Plan a 3-day trip to Auckland")
        assert params["days"] == 3

    def test_destination_parsing(self) -> None:
        """Verify destination is parsed correctly."""
        params = parse_trip_request("Plan a trip to Queenstown for $1000")
        assert params["destination"] == "Queenstown"

    def test_budget_constraint_in_hotels(self) -> None:
        """Verify hotels filter respects max_price constraint."""
        hotels = get_hotels("Auckland", "2026-02-20", "2026-02-22", max_price=100)
        
        # All returned hotels should be under max_price
        for hotel in hotels:
            assert hotel.price_per_night <= 100

    def test_flight_prices_realistic(self) -> None:
        """Verify flight prices are within realistic range."""
        flights = get_flights("Wellington", "Auckland", "2026-02-20")
        
        for flight in flights:
            # NZ domestic flights typically $50-$250
            assert 50 <= flight.price <= 250


class TestJsonSchema:
    """Tests for JSON schema validation."""

    def test_itinerary_schema_valid(self) -> None:
        """Verify Itinerary schema accepts valid data."""
        itinerary = Itinerary(
            destination="Auckland",
            duration_days=2,
            total_cost_nzd=450.0,
            budget_nzd=500.0,
            within_budget=True,
            flights=[],
            hotel=None,
            itinerary=[
                ItineraryDay(
                    day=1,
                    date="2026-02-20",
                    activities=["Sky Tower", "Viaduct Harbour"],
                    accommodation="CityLife Auckland",
                    day_cost=200.0,
                ),
                ItineraryDay(
                    day=2,
                    date="2026-02-21",
                    activities=["Auckland Museum", "Mount Eden"],
                    accommodation="CityLife Auckland",
                    day_cost=250.0,
                ),
            ],
        )
        
        assert itinerary.destination == "Auckland"
        assert itinerary.duration_days == 2
        assert itinerary.within_budget is True
        assert len(itinerary.itinerary) == 2

    def test_itinerary_schema_to_json(self) -> None:
        """Verify Itinerary can be serialized to JSON."""
        itinerary = Itinerary(
            destination="Auckland",
            duration_days=2,
            total_cost_nzd=450.0,
            budget_nzd=500.0,
            within_budget=True,
            flights=[],
            hotel=None,
            itinerary=[],
        )
        
        # Should not raise
        json_str = itinerary.model_dump_json()
        parsed = json.loads(json_str)
        
        assert parsed["destination"] == "Auckland"
        assert parsed["total_cost_nzd"] == 450.0

    def test_flight_result_schema(self) -> None:
        """Verify FlightResult schema."""
        flight = FlightResult(
            airline="Air New Zealand",
            flight_number="NZ123",
            price=149.0,
            departure="08:00",
            arrival="09:15",
            origin="Wellington",
            destination="Auckland",
        )
        
        assert flight.airline == "Air New Zealand"
        assert flight.price == 149.0

    def test_hotel_result_schema(self) -> None:
        """Verify HotelResult schema with rating bounds."""
        hotel = HotelResult(
            name="Test Hotel",
            price_per_night=150.0,
            rating=4.5,
            address="123 Test St",
            amenities=["WiFi", "Pool"],
        )
        
        assert hotel.rating == 4.5
        assert "WiFi" in hotel.amenities

    def test_hotel_rating_bounds(self) -> None:
        """Verify hotel rating must be 0-5."""
        with pytest.raises(ValidationError):
            HotelResult(
                name="Test Hotel",
                price_per_night=150.0,
                rating=6.0,  # Invalid - above 5
                address="123 Test St",
                amenities=[],
            )

    def test_weather_result_schema(self) -> None:
        """Verify WeatherResult schema."""
        weather = WeatherResult(
            date="2026-02-20",
            condition="Sunny",
            temp_high=25.0,
            temp_low=18.0,
            precipitation_chance=10,
        )
        
        assert weather.condition == "Sunny"
        assert weather.temp_high > weather.temp_low

    def test_attraction_result_schema(self) -> None:
        """Verify AttractionResult schema."""
        attraction = AttractionResult(
            name="Sky Tower",
            category="landmark",
            price=32.0,
            rating=4.5,
            description="Iconic tower",
            duration_hours=1.5,
        )
        
        assert attraction.name == "Sky Tower"
        assert attraction.price == 32.0


class TestToolFunctions:
    """Tests for mock tool functions."""

    def test_get_flights_returns_list(self) -> None:
        """Verify get_flights returns a list of FlightResult."""
        flights = get_flights("Wellington", "Auckland", "2026-02-20")
        
        assert isinstance(flights, list)
        assert len(flights) >= 1
        assert all(isinstance(f, FlightResult) for f in flights)

    def test_get_hotels_returns_list(self) -> None:
        """Verify get_hotels returns a list of HotelResult."""
        hotels = get_hotels("Auckland", "2026-02-20", "2026-02-22")
        
        assert isinstance(hotels, list)
        assert len(hotels) >= 1
        assert all(isinstance(h, HotelResult) for h in hotels)

    def test_get_weather_returns_result(self) -> None:
        """Verify get_weather returns a WeatherResult."""
        weather = get_weather("Auckland", "2026-02-20")
        
        assert isinstance(weather, WeatherResult)
        assert weather.date == "2026-02-20"

    def test_get_attractions_returns_list(self) -> None:
        """Verify get_attractions returns a list of AttractionResult."""
        attractions = get_attractions("Auckland")
        
        assert isinstance(attractions, list)
        assert len(attractions) >= 1
        assert all(isinstance(a, AttractionResult) for a in attractions)

    def test_flights_sorted_by_price(self) -> None:
        """Verify flights are returned sorted by price."""
        flights = get_flights("Wellington", "Auckland", "2026-02-20")
        
        prices = [f.price for f in flights]
        assert prices == sorted(prices)

    def test_hotels_sorted_by_price(self) -> None:
        """Verify hotels are returned sorted by price."""
        hotels = get_hotels("Auckland", "2026-02-20", "2026-02-22")
        
        prices = [h.price_per_night for h in hotels]
        assert prices == sorted(prices)
