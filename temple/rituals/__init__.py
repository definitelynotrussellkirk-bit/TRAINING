"""
Temple Rituals - Available diagnostic ritual implementations.

Each ritual module defines checks using the @register_ritual decorator.
Rituals are automatically loaded when temple.list_rituals() or
temple.run_ritual() is called.

Available rituals:
- quick: Fast sanity checks on core services
- api: Detailed HTTP API endpoint validation
"""
