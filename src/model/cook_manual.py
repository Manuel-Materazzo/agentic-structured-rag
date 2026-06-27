from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class License(BaseModel):
    license_type: Literal['p', 't', 'g', 'e+', 'mx', 'q', 'c', 'ltk'] = Field(
        ...,
        description="The specific category code for the license."
    )
    license_grade: int = Field(
        ...,
        description="The grade level, stored as an integer (e.g., 6 instead of 'VI')."
    )

    @field_validator('license_grade', mode='before')
    @classmethod
    def convert_roman_to_int(cls, value):
        """
        Automatically converts basic Roman numerals to integers
        if the raw data comes in as a string.
        """
        if isinstance(value, str):
            roman_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
            upper_val = value.upper().strip()
            if upper_val in roman_map:
                return roman_map[upper_val]

            # Fallback if it's a numeric string like "6"
            if upper_val.isdigit():
                return int(upper_val)

        return value


class Technique(BaseModel):
    name: str = Field(..., description="The unique name of the technique.")
    macro_category: str = Field(..., description="The broader category this technique falls under.")
    licenses: List[License] = Field(default_factory=list, description="List of required licenses.")


class CookManualParsingResult(BaseModel):
    techniques: List[Technique] = Field(default_factory=list, description="List of processed techniques.")
    parsing_confidence: Literal['high', 'low'] = Field(..., description="Confidence score of the parser.")
    parsing_issues: Optional[str] = Field(
        None,
        description="Description of errors/warnings encountered, or null if seamless."
    )
