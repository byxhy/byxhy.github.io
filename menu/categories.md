---
layout: page
title: Categories
---
<ul class="posts">
  {% for post in site.posts %}

    {% unless post.next %}
      <h3>{{ post.categories}}</h3>
    {% else %}
      {% capture category %}{{ post.categories}}{% endcapture %}
      {% capture ncategory %}{{ post.next.categories}}{% endcapture %}
      {% if category != ncategory %}
        <h3>{{ post.categories | categories: '%Y' }}</h3>
      {% endif %}
    {% endunless %}

    <li itemscope>
      <a href="{{ site.github.url }}{{ post.url }}">{{ post.title }}</a>
      <p class="post-date"><span><i class="fa fa-calendar" aria-hidden="true"></i> {{ post.date | date: "%B %-d" }} - <i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}</span></p>
    </li>

  {% endfor %}
</ul>
