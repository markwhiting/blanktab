// Everyday vocabulary for generated content. Each group is a space-separated list; all words are lowercase a-z, 2-10 letters.
const group = (s: string): string[] => s.split(/\s+/).filter(Boolean)

const ANIMALS = group(`cat dog fish bird horse cow pig sheep goat duck hen rabbit mouse frog bear lion tiger
  wolf fox deer monkey zebra snake turtle whale shark seal crab bee ant spider worm snail owl eagle
  parrot camel donkey kitten puppy lamb squirrel hedgehog otter beaver`)

const COLOURS = group(`red blue green yellow orange purple pink brown black white grey gold silver
  violet beige cream tan navy teal lime maroon`)

const BODY = group(`head hair face eye ear nose mouth tooth tongue lip neck arm hand finger thumb leg foot
  toe knee elbow back chest belly skin bone heart brain shoulder ankle wrist cheek chin`)

const HOME = group(`house room door window wall floor roof table chair bed sofa lamp clock mirror shelf
  desk stairs garden fence kitchen bath cup plate bowl spoon fork knife pan pot bottle box bag basket
  blanket pillow carpet curtain key lock broom bucket`)

const FOOD = group(`bread milk egg cheese butter apple banana grape lemon peach pear plum cake pie cookie
  candy sugar salt soup rice pasta pizza meat chicken bean carrot potato onion tomato corn pepper honey
  juice tea water jam toast cereal berry melon`)

const CLOTHES = group(`hat cap coat shirt shoe sock boot glove scarf dress skirt pants jacket belt button
  pocket sweater jeans sandal apron ribbon`)

const NATURE = group(`sun moon star sky cloud rain snow wind storm river lake sea ocean beach sand rock
  stone hill mountain valley forest tree leaf flower grass seed root branch field cave island fire ice
  mud wave rainbow thunder`)

const PLACES = group(`home school shop store market park farm town city village street road bridge church
  library museum hospital station airport harbor castle tower office barn zoo cafe hotel`)

const TIME = group(`day night morning evening noon week month year today tomorrow yesterday hour minute
  second spring summer autumn winter monday friday sunday early late soon`)

const NUMBERS = group(`one two three four five six seven eight nine ten eleven twelve twenty hundred
  thousand first second third half double pair`)

const PEOPLE = group(`mother father sister brother baby child boy girl man woman friend teacher doctor
  farmer baker king queen nurse pilot sailor driver artist singer neighbor uncle aunt cousin grandma`)

const THINGS = group(`ball doll toy block puzzle kite drum bell book paper pencil pen card map picture
  letter coin ring boat ship train bus car truck bike plane wheel rope net tent flag crown drawing
  game song story music hammer paint brush`)

const VERBS = group(`run walk jump sit stand sleep eat drink play read write draw sing dance laugh cry
  smile look listen speak talk think know learn teach help give take make build open close push pull
  throw catch carry climb swim fly ride drive cook wash clean cut fold hide seek find lose win want
  need like love hold turn move stop start wait watch show tell ask call come go bring put`)

const ADJECTIVES = group(`big small tall short long wide thin thick heavy light fast slow hot cold warm
  cool wet dry soft hard loud quiet happy sad angry kind brave shy clever silly new old young fresh
  clean dirty empty full round square sharp smooth rough bright dark sweet sour sticky shiny`)

const MISC = group(`yes no maybe up down left right near far here there inside outside above below under
  over front behind between around always never often again together alone many few some all each
  every more less most same other next last high low deep`)

const WORD = /^[a-z]{2,10}$/

/** Returns `list` unchanged, or throws on the first entry that is not 2-10 lowercase a-z letters. */
export function validateWords(list: readonly string[]): readonly string[] {
  for (const w of list) if (!WORD.test(w)) throw new Error(`invalid word ${JSON.stringify(w)}: must match ${WORD}`)
  return list
}

// Validated at module load: a bad entry fails loudly rather than vanishing from the vocabulary.
export const WORDS: readonly string[] = validateWords([...new Set([
  ...ANIMALS, ...COLOURS, ...BODY, ...HOME, ...FOOD, ...CLOTHES, ...NATURE, ...PLACES, ...TIME,
  ...NUMBERS, ...PEOPLE, ...THINGS, ...VERBS, ...ADJECTIVES, ...MISC,
])])
