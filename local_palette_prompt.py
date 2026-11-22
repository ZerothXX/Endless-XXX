"""Deterministic palette wording without encoding a swatch layout into pixels."""
import colorsys

def describe_color(rgb):
    r,g,b=[x/255 for x in rgb];h,s,v=colorsys.rgb_to_hsv(r,g,b);h*=360
    if s<.12:
        return 'off-white' if v>.85 else 'light gray' if v>.65 else 'gray' if v>.35 else 'charcoal' if v>.13 else 'near-black'
    hue=('red' if h<15 or h>=345 else 'orange' if h<45 else 'yellow' if h<70 else 'green' if h<165 else
         'cyan' if h<195 else 'blue' if h<255 else 'violet' if h<285 else 'pink' if h<345 else 'red')
    shade='pale ' if v>.85 and s<.35 else 'light ' if v>.75 and s<.5 else 'deep ' if v<.45 else ''
    muted='muted ' if s<.35 and not shade else ''
    return shade+muted+hue

def palette_instruction(palette):
    colors=', '.join(f"{describe_color(x['rgb'])} (#{''.join(f'{int(v):02X}' for v in x['rgb'])}, about {round(100*x['fraction'])}% of clothing-color evidence)" for x in palette)
    return ('The automatically measured clothing palette is: '+colors+'. '
        'Use these as harmonious material colors, not as a diagram, chart, equal-width stripes or a literal percentage layout. '
        'Use isolated motif-reference colors for small accents. Preserve the target material, light and functional form. ')
