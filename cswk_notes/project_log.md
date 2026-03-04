
## 4 March

initial 1.1 and 1.2 by claude, but got fooled when there were faces in the background (eg. this one below was classified as full body front because of the full body detection then the face ove the right shoulder)![[Pasted image 20260304175909.png]]
![[Pasted image 20260304175429.png]]
![[Pasted image 20260304175909.png]]
⇒ changed logic to look fo face WITHIN the yolo bounding box (Rather than the whole image)
then this got correctly classified as full body back

But still some errors for side on pictures:
![[Pasted image 20260304180654.png]]
- it spots 1 ear so model naively assumed this means its facing front