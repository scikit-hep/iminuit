from iminuit.color import Gradient


def test_color_2():
    g = Gradient((-1, 10, 10, 20), (2, 20, 20, 10))
    assert g.rgb(-1) == "rgb(10,10,20)"
    assert g.rgb(2) == "rgb(20,20,10)"
    assert g.rgb(-1.00001) == "rgb(10,10,20)"
    assert g.rgb(1.99999) == "rgb(20,20,10)"
    assert g.rgb(0.5) == "rgb(15,15,15)"


def test_color_3():
    g = Gradient((-1, 50, 50, 250), (0, 100, 100, 100), (1, 250, 50, 50))
    assert g.rgb(-1) == "rgb(50,50,250)"
    assert g.rgb(-0.5) == "rgb(75,75,175)"
    assert g.rgb(0) == "rgb(100,100,100)"
    assert g.rgb(0.5) == "rgb(175,75,75)"
    assert g.rgb(1) == "rgb(250,50,50)"
