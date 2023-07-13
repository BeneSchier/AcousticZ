from AcousticZ.Room import Room

def test_room_creation():
    room = Room()
    assert room.is_empty()