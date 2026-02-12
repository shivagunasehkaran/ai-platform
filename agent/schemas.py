"""Pydantic schemas for agent tool inputs and outputs."""

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a tool call from the LLM."""
    name: str = Field(..., description="Name of the tool to call")
    arguments: dict = Field(default_factory=dict, description="Arguments for the tool")


class FlightResult(BaseModel):
    """Flight search result."""
    airline: str = Field(..., description="Airline name")
    flight_number: str = Field(..., description="Flight number")
    price: float = Field(..., description="Price in NZD")
    departure: str = Field(..., description="Departure time")
    arrival: str = Field(..., description="Arrival time")
    origin: str = Field(..., description="Origin airport")
    destination: str = Field(..., description="Destination airport")


class HotelResult(BaseModel):
    """Hotel search result."""
    name: str = Field(..., description="Hotel name")
    price_per_night: float = Field(..., description="Price per night in NZD")
    rating: float = Field(..., ge=0, le=5, description="Hotel rating (0-5)")
    address: str = Field(..., description="Hotel address")
    amenities: list[str] = Field(default_factory=list, description="Hotel amenities")


class WeatherResult(BaseModel):
    """Weather forecast result."""
    date: str = Field(..., description="Date of forecast")
    condition: str = Field(..., description="Weather condition")
    temp_high: float = Field(..., description="High temperature in Celsius")
    temp_low: float = Field(..., description="Low temperature in Celsius")
    precipitation_chance: int = Field(..., description="Chance of precipitation (%)")


class AttractionResult(BaseModel):
    """Tourist attraction result."""
    name: str = Field(..., description="Attraction name")
    category: str = Field(..., description="Category (museum, park, landmark, etc.)")
    price: float = Field(..., description="Entry price in NZD (0 if free)")
    rating: float = Field(..., ge=0, le=5, description="Rating (0-5)")
    description: str = Field(..., description="Brief description")
    duration_hours: float = Field(..., description="Typical visit duration in hours")


class ItineraryDay(BaseModel):
    """Single day in an itinerary."""
    day: int = Field(..., description="Day number (1, 2, 3, ...)")
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    activities: list[str] = Field(..., description="List of activities for the day")
    accommodation: str = Field(..., description="Hotel name for overnight stay")
    day_cost: float = Field(..., description="Total cost for this day in NZD")


class Itinerary(BaseModel):
    """Complete travel itinerary."""
    destination: str = Field(..., description="Destination city")
    duration_days: int = Field(..., description="Number of days")
    total_cost_nzd: float = Field(..., description="Total trip cost in NZD")
    budget_nzd: float = Field(..., description="Original budget in NZD")
    within_budget: bool = Field(..., description="Whether trip is within budget")
    flights: list[FlightResult] = Field(default_factory=list, description="Booked flights")
    hotel: HotelResult | None = Field(None, description="Selected hotel")
    itinerary: list[ItineraryDay] = Field(..., description="Day-by-day itinerary")


class TripConstraints(BaseModel):
    """Parsed trip constraints from user prompt."""
    destination: str = Field(..., description="Destination city")
    origin: str = Field(default="Wellington", description="Origin city")
    duration_days: int = Field(..., description="Number of days")
    budget_nzd: float = Field(..., description="Maximum budget in NZD")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
