from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Restaurant(BaseModel):
    name: str = Field(
        description="The name of the restaurant."
    )
    chef: Optional[str] = Field(
        default=None,
        description="The name of the head chef, or null if unknown."
    )
    planet: Optional[str] = Field(
        default=None,
        description="The planet where the restaurant is located, or null if unknown."
    )
    professional_orders: List[str] = Field(
        description="A list of professional associations or culinary orders the restaurant belongs to. Leave empty if none."
    )


class Ingredient(BaseModel):
    name: str = Field(
        description="The specific name of the ingredient."
    )
    quantity_grams: Optional[float] = Field(
        default=None,
        description="The calculated or parsed weight of the ingredient in grams. Use null if it cannot be converted to grams."
    )
    quantity_raw: str = Field(
        description="The verbatim text describing the quantity as written in the original source text."
    )


class Dish(BaseModel):
    name: str = Field(
        description="The name of the dish."
    )
    ingredients: List[Ingredient] = Field(
        description="The list of ingredients required for this dish."
    )
    techniques: List[str] = Field(
        description="Specific culinary techniques used to prepare the dish (e.g., 'sous-vide', 'searing', 'fermentation')."
    )
    preparation_notes: Optional[str] = Field(
        default=None,
        description="Additional text, context, or instructions regarding preparation, or null if none."
    )


class License(BaseModel):
    license_type: str = Field(
        description="Type of license held. Must strictly map to one of these shortcodes: 'p' (Psionica), 't' (Temporale), 'g' (Gravitazionale), 'e+' (Antimateria), 'mx' (Magnetica), 'q' (Quantistica), 'c' (Luce), 'ltk' (Livello di Sviluppo Tecnologico)"
    )
    license_grade: int = Field(
        description="License grade (converted integer in case of roman numerals, e.g. 'VI -> 6')"
    )


class RestaurantData(BaseModel):
    restaurant: Restaurant = Field(
        description="Details about the dining establishment."
    )
    dishes: List[Dish] = Field(
        description="A list of dishes offered by the restaurant."
    )
    licenses: List[License] = Field(
        description="A list of the restaurant chef's professional culinary licenses."
    )
    parsing_confidence: Literal["high", "low"] = Field(
        description="Must be exactly 'high' or 'low'. Use 'high' if all critical data was successfully extracted, and 'low' if key details were missing, ambiguous, or failed calculation."
    )
    parsing_issues: Optional[str] = Field(
        default=None,
        description="A text description of any errors, unresolvable quantities, missing fields, or conflicts encountered during parsing. Use null if parsing was entirely successful."
    )
