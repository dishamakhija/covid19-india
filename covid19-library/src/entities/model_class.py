import enum


@enum.unique
class ModelClass(str, enum.Enum):
    SEIR = "SEIR"
    IHME = "IHME"
    SEIHRD = "SEIHRD"
    SEIR_gen = "SEIR_gen"
