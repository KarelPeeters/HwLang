use crate::syntax::format::FormatSettings;
use crate::syntax::format_new::low::{LCommaGroup, LNode};
use crate::syntax::pos::LineOffsets;
use crate::syntax::token::Token;
use crate::util::iter::IterExt;
use crate::util::{Never, ResultNeverExt};
use itertools::Itertools;
